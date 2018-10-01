//! Scoped thread.
//!
//! # Examples
//!
//! A basic scoped thread:
//!
//! ```
//! crossbeam_utils::thread::scope(|scope| {
//!     scope.spawn(|_| {
//!         println!("Hello from a scoped thread!");
//!     });
//! }).unwrap();
//! ```
//!
//! When writing concurrent Rust programs, you'll sometimes see a pattern like this, using
//! [`std::thread::spawn`]:
//!
//! ```ignore
//! let array = [1, 2, 3];
//! let mut guards = vec![];
//!
//! for i in &array {
//!     let guard = std::thread::spawn(move || {
//!         println!("element: {}", i);
//!     });
//!
//!     guards.push(guard);
//! }
//!
//! for guard in guards {
//!     guard.join().unwrap();
//! }
//! ```
//!
//! The basic pattern is:
//!
//! 1. Iterate over some collection.
//! 2. Spin up a thread to operate on each part of the collection.
//! 3. Join all the threads.
//!
//! However, this code actually gives an error:
//!
//! ```text
//! error: `array` does not live long enough
//! for i in &array {
//!           ^~~~~
//! in expansion of for loop expansion
//! note: expansion site
//! note: reference must be valid for the static lifetime...
//! note: ...but borrowed value is only valid for the block suffix following statement 0 at ...
//!     let array = [1, 2, 3];
//!     let mut guards = vec![];
//!
//!     for i in &array {
//!         let guard = std::thread::spawn(move || {
//!             println!("element: {}", i);
//! ...
//! error: aborting due to previous error
//! ```
//!
//! Because [`std::thread::spawn`] doesn't know about this scope, it requires a `'static` lifetime.
//! One way of giving it a proper lifetime is to use an [`Arc`]:
//!
//! [`Arc`]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html
//! [`std::thread::spawn`]: https://doc.rust-lang.org/stable/std/thread/fn.spawn.html
//!
//! ```
//! use std::sync::Arc;
//!
//! let array = Arc::new([1, 2, 3]);
//! let mut guards = vec![];
//!
//! for i in 0..array.len() {
//!     let a = array.clone();
//!
//!     let guard = std::thread::spawn(move || {
//!         println!("element: {}", a[i]);
//!     });
//!
//!     guards.push(guard);
//! }
//!
//! for guard in guards {
//!     guard.join().unwrap();
//! }
//! ```
//!
//! But this introduces unnecessary allocation, as `Arc<T>` puts its data on the heap, and we
//! also end up dealing with reference counts. We know that we're joining the threads before
//! our function returns, so just taking a reference _should_ be safe. Rust can't know that,
//! though.
//!
//! Enter scoped threads. Here's our original example, using `spawn` from crossbeam rather
//! than from `std::thread`:
//!
//! ```
//! let array = [1, 2, 3];
//!
//! crossbeam_utils::thread::scope(|scope| {
//!     for i in &array {
//!         scope.spawn(move |_| {
//!             println!("element: {}", i);
//!         });
//!     }
//! }).unwrap();
//! ```
//!
//! Much more straightforward.

use std::fmt;
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::panic;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

/// Like [`std::thread::spawn`], but without lifetime bounds on the closure.
///
/// [`std::thread::spawn`]: https://doc.rust-lang.org/stable/std/thread/fn.spawn.html
pub unsafe fn spawn_unchecked<'env, F, T>(f: F) -> thread::JoinHandle<T>
where
    F: FnOnce() -> T,
    F: Send + 'env,
    T: Send + 'static,
{
    let builder = thread::Builder::new();
    builder_spawn_unchecked(builder, f).unwrap()
}

/// Like [`std::thread::Builder::spawn`], but without lifetime bounds on the closure.
///
/// [`std::thread::Builder::spawn`]:
///     https://doc.rust-lang.org/nightly/std/thread/struct.Builder.html#method.spawn
pub unsafe fn builder_spawn_unchecked<'env, F, T>(
    builder: thread::Builder,
    f: F,
) -> io::Result<thread::JoinHandle<T>>
where
    F: FnOnce() -> T,
    F: Send + 'env,
    T: Send + 'static,
{
    // Change the type of `f` from `FnOnce() -> T` to `FnMut() -> T`.
    let mut f = Some(f);
    let f = move || f.take().unwrap()();

    // Allocate it on the heap and erase the `'env` bound.
    let f: Box<FnMut() -> T + Send + 'env> = Box::new(f);
    let f: Box<FnMut() -> T + Send + 'static> = mem::transmute(f);

    // Finally, spawn the closure.
    let mut f = f;
    builder.spawn(move || f())
}

/// Creates a new `Scope` for [*scoped thread spawning*](struct.Scope.html#method.spawn).
///
/// No matter what happens, before the `Scope` is dropped, it is guaranteed that all the unjoined
/// spawned scoped threads are joined.
///
/// `thread::scope()` returns `Ok(())` if all the unjoined spawned threads did not panic. It returns
/// `Err(e)` if one of them panics with `e`. If many of them panic, it is still guaranteed that all
/// the threads are joined, and `thread::scope()` returns `Err(e)` with `e` from a panicking thread.
///
/// # Examples
///
/// Creating and using a scope:
///
/// ```
/// crossbeam_utils::thread::scope(|scope| {
///     scope.spawn(|_| println!("Exiting scope"));
///     scope.spawn(|_| println!("Running child thread in scope"));
/// }).unwrap();
/// ```
pub fn scope<'env, F, R>(f: F) -> thread::Result<R>
where
    F: FnOnce(&Scope<'env>) -> R,
{
    let (tx, rx) = mpsc::channel();
    let scope = Scope {
        handles: Arc::new(Mutex::new(Vec::new())),
        chan: tx,
        _marker: PhantomData,
    };

    // Execute the scoped function, but catch any panics.
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(&scope)));

    // Wait until all nested scopes are dropped.
    drop(scope.chan);
    let _ = rx.recv();

    // Join all remaining spawned threads.
    let panics: Vec<_> = {
        let mut handles = scope
            .handles
            .lock()
            .unwrap();

        // Filter handles that haven't been joined, join them, and collect errors.
        let panics = handles
            .drain(..)
            .filter_map(|handle| handle.lock().unwrap().take())
            .filter_map(|handle| handle.join().err())
            .collect();

        panics
    };

    // If `f` has panicked, resume unwinding.
    // If any of the child threads have panicked, return the panic errors.
    // Otherwise, everything is OK and return the result of `f`.
    match result {
        Err(err) => panic::resume_unwind(err),
        Ok(res) => {
            if panics.is_empty() {
                Ok(res)
            } else {
                Err(Box::new(panics))
            }
        }
    }
}

pub struct Scope<'env> {
    /// The list of the thread join handles.
    handles: Arc<Mutex<Vec<Arc<Mutex<Option<thread::JoinHandle<()>>>>>>>,

    /// Dropping this sender is a signal that the scope has been dropped.
    chan: mpsc::Sender<()>,

    /// Borrows data with lifetime `'env`.
    _marker: PhantomData<&'env ()>,
}

unsafe impl<'env> Sync for Scope<'env> {}

impl<'env> Scope<'env> {
    /// Create a scoped thread.
    ///
    /// `spawn` is similar to the [`spawn`] function in Rust's standard library. The difference is
    /// that this thread is scoped, meaning that it's guaranteed to terminate before the current
    /// stack frame goes away, allowing you to reference the parent stack frame directly. This is
    /// ensured by having the parent thread join on the child thread before the scope exits.
    ///
    /// [`spawn`]: https://doc.rust-lang.org/std/thread/fn.spawn.html
    pub fn spawn<'scope, F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce(&Scope<'env>) -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        self.builder().spawn(f).unwrap()
    }

    /// Generates the base configuration for spawning a scoped thread, from which configuration
    /// methods can be chained.
    pub fn builder<'scope>(&'scope self) -> ScopedThreadBuilder<'scope, 'env> {
        ScopedThreadBuilder {
            scope: self,
            builder: thread::Builder::new(),
        }
    }
}

impl<'env> fmt::Debug for Scope<'env> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scope {{ ... }}")
    }
}

/// Scoped thread configuration. Provides detailed control over the properties and behavior of new
/// scoped threads.
pub struct ScopedThreadBuilder<'scope, 'env: 'scope> {
    scope: &'scope Scope<'env>,
    builder: thread::Builder,
}

impl<'scope, 'env> ScopedThreadBuilder<'scope, 'env> {
    /// Names the thread-to-be. Currently the name is used for identification only in panic
    /// messages.
    pub fn name(mut self, name: String) -> ScopedThreadBuilder<'scope, 'env> {
        self.builder = self.builder.name(name);
        self
    }

    /// Sets the size of the stack for the new thread.
    pub fn stack_size(mut self, size: usize) -> ScopedThreadBuilder<'scope, 'env> {
        self.builder = self.builder.stack_size(size);
        self
    }

    /// Spawns a new thread, and returns a join handle for it.
    pub fn spawn<F, T>(self, f: F) -> io::Result<ScopedJoinHandle<'scope, T>>
    where
        F: FnOnce(&Scope<'env>) -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        // The result of `f` will be stored here.
        let result = Arc::new(Mutex::new(None));

        // Spawn the thread and grab its join handle and thread handle.
        let (handle, thread) = {
            let result = Arc::clone(&result);

            // A clone of the scope that will be moved into the new thread.
            let scope = Scope {
                handles: Arc::clone(&self.scope.handles),
                chan: self.scope.chan.clone(),
                _marker: PhantomData,
            };

            // Spawn the thread.
            let handle = unsafe {
                builder_spawn_unchecked(self.builder, move || {
                    // Make sure the scope is inside the closure with the proper `'env` lifetime.
                    let scope: Scope<'env> = scope;

                    // Run the closure.
                    let res = f(&scope);

                    // Store the result if the closure didn't panic.
                    *result.lock().unwrap() = Some(res);
                })?
            };

            let thread = handle.thread().clone();
            let handle = Arc::new(Mutex::new(Some(handle)));
            (handle, thread)
        };

        // Add the handle to the shared list of join handles.
        self.scope
            .handles
            .lock()
            .unwrap()
            .push(Arc::clone(&handle));

        Ok(ScopedJoinHandle {
            handle,
            result,
            thread,
            _marker: PhantomData,
        })
    }
}

unsafe impl<'scope, T> Send for ScopedJoinHandle<'scope, T> {}
unsafe impl<'scope, T> Sync for ScopedJoinHandle<'scope, T> {}

/// A handle to a scoped thread
pub struct ScopedJoinHandle<'scope, T: 'scope> {
    /// A join handle to the spawned thread.
    handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,

    /// Holds the result of the inner closure.
    result: Arc<Mutex<Option<T>>>,

    /// A handle to the the spawned thread.
    thread: thread::Thread,

    /// Borrows the parent scope with lifetime `'scope`.
    _marker: PhantomData<&'scope T>,
}

impl<'scope, T> ScopedJoinHandle<'scope, T> {
    /// Waits for the associated thread to finish.
    ///
    /// If the child thread panics, [`Err`] is returned with the parameter given to [`panic`].
    ///
    /// [`Err`]: https://doc.rust-lang.org/std/result/enum.Result.html#variant.Err
    /// [`panic`]: https://doc.rust-lang.org/std/macro.panic.html
    ///
    /// # Panics
    ///
    /// This function may panic on some platforms if a thread attempts to join itself or otherwise
    /// may create a deadlock with joining threads.
    pub fn join(self) -> thread::Result<T> {
        // Take out the handle. The handle will surely be available because the root scope waits
        // for nested scopes before joining remaining threads.
        let handle = self
            .handle
            .lock()
            .unwrap()
            .take()
            .unwrap();

        // Join the thread and then take the result out of its inner closure.
        handle.join().map(|()| {
            self.result
                .lock()
                .unwrap()
                .take()
                .unwrap()
        })
    }

    /// Gets the underlying [`std::thread::Thread`] handle.
    ///
    /// [`std::thread::Thread`]: https://doc.rust-lang.org/std/thread/struct.Thread.html
    pub fn thread(&self) -> &thread::Thread {
        &self.thread
    }
}

impl<'scope, T> fmt::Debug for ScopedJoinHandle<'scope, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ScopedJoinHandle {{ ... }}")
    }
}
