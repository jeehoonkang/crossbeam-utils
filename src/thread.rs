/// Scoped thread.
///
/// # Examples
///
/// A basic scoped thread:
///
/// ```
/// crossbeam_utils::thread::scope(|scope| {
///     scope.spawn(|| {
///         println!("Hello from a scoped thread!");
///     });
/// }).unwrap();
/// ```
///
/// When writing concurrent Rust programs, you'll sometimes see a pattern like this, using
/// [`std::thread::spawn`]:
///
/// ```ignore
/// let array = [1, 2, 3];
/// let mut guards = vec![];
///
/// for i in &array {
///     let guard = std::thread::spawn(move || {
///         println!("element: {}", i);
///     });
///
///     guards.push(guard);
/// }
///
/// for guard in guards {
///     guard.join().unwrap();
/// }
/// ```
///
/// The basic pattern is:
///
/// 1. Iterate over some collection.
/// 2. Spin up a thread to operate on each part of the collection.
/// 3. Join all the threads.
///
/// However, this code actually gives an error:
///
/// ```text
/// error: `array` does not live long enough
/// for i in &array {
///           ^~~~~
/// in expansion of for loop expansion
/// note: expansion site
/// note: reference must be valid for the static lifetime...
/// note: ...but borrowed value is only valid for the block suffix following statement 0 at ...
///     let array = [1, 2, 3];
///     let mut guards = vec![];
///
///     for i in &array {
///         let guard = std::thread::spawn(move || {
///             println!("element: {}", i);
/// ...
/// error: aborting due to previous error
/// ```
///
/// Because [`std::thread::spawn`] doesn't know about this scope, it requires a `'static` lifetime.
/// One way of giving it a proper lifetime is to use an [`Arc`]:
///
/// [`Arc`]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html
/// [`std::thread::spawn`]: https://doc.rust-lang.org/stable/std/thread/fn.spawn.html
///
/// ```
/// use std::sync::Arc;
///
/// let array = Arc::new([1, 2, 3]);
/// let mut guards = vec![];
///
/// for i in 0..array.len() {
///     let a = array.clone();
///
///     let guard = std::thread::spawn(move || {
///         println!("element: {}", a[i]);
///     });
///
///     guards.push(guard);
/// }
///
/// for guard in guards {
///     guard.join().unwrap();
/// }
/// ```
///
/// But this introduces unnecessary allocation, as `Arc<T>` puts its data on the heap, and we
/// also end up dealing with reference counts. We know that we're joining the threads before
/// our function returns, so just taking a reference _should_ be safe. Rust can't know that,
/// though.
///
/// Enter scoped threads. Here's our original example, using `spawn` from crossbeam rather
/// than from `std::thread`:
///
/// ```
/// let array = [1, 2, 3];
///
/// crossbeam_utils::thread::scope(|scope| {
///     for i in &array {
///         scope.spawn(move || {
///             println!("element: {}", i);
///         });
///     }
/// }).unwrap();
/// ```
///
/// Much more straightforward.
use std::cell::RefCell;
use std::fmt;
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::panic;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;

/// Like [`std::thread::spawn`], but without lifetime bounds on the closure.
///
/// [`std::thread::spawn`]: https://doc.rust-lang.org/stable/std/thread/fn.spawn.html
pub unsafe fn spawn_unchecked<'a, F, T>(f: F) -> thread::JoinHandle<T>
where
    F: FnOnce() -> T,
    F: Send + 'a,
    T: Send + 'static,
{
    let builder = thread::Builder::new();
    builder_spawn_unchecked(builder, f).unwrap()
}

/// Like [`std::thread::Builder::spawn`], but without lifetime bounds on the closure.
///
/// [`std::thread::Builder::spawn`]:
///     https://doc.rust-lang.org/nightly/std/thread/struct.Builder.html#method.spawn
pub unsafe fn builder_spawn_unchecked<'a, F, T>(
    builder: thread::Builder,
    f: F,
) -> io::Result<thread::JoinHandle<T>>
where
    F: FnOnce() -> T,
    F: Send + 'a,
    T: Send + 'static,
{
    let closure: Box<FnBox<T> + 'a> = Box::new(f);
    let closure: Box<FnBox<T> + Send> = mem::transmute(closure);
    builder.spawn(move || closure.call_box())
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
///     scope.spawn(|| println!("Exiting scope"));
///     scope.spawn(|| println!("Running child thread in scope"));
/// }).unwrap();
/// ```
pub fn scope<'env, F, R>(f: F) -> thread::Result<R>
where
    F: FnOnce(&Scope<'env>) -> R,
{
    let scope = Default::default();

    // Executes the scoped function. Panics will be caught as `Err`, which makes this function
    // exception safe.
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(&scope)));

    // Joins all the threads, if any of the threads (joins) panicked, returns the error Result.
    scope.join_all()?;
    result
}

#[derive(Default)]
pub struct Scope<'env> {
    /// The list of the thread join jobs.
    joins: RefCell<Vec<Box<FnBox<thread::Result<()>> + 'env>>>,
    // !Send + !Sync
    _marker: PhantomData<*const ()>,
}

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
        F: FnOnce() -> T,
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

    /// Generates a wrapper around the scope that enables stat counting on all newly spawned or
    /// built scoped threads.
    pub fn stat_counting<'scope>(&'scope self) -> StatCountingScope<'scope, 'env> {
        StatCountingScope::from(self)
    }

    // This method is carefully written in a transactional style, so that it can be called directly
    // and, if any thread join panics, can be resumed in the unwinding this causes. By initially
    // running the method outside of any destructor, we avoid any leakage problems due to
    // @rust-lang/rust#14875.
    //
    // FIXME(jeehoonkang): @rust-lang/rust#14875 is fixed, so maybe we can remove the above comment.
    // But I'd like to write tests to check it before removing the comment.
    fn join_all(self) -> thread::Result<()> {
        self.joins
            .into_inner()
            .into_iter()
            .fold(Ok(()), |result, join| result.and(join.call_box()))
    }
}

impl<'env> fmt::Debug for Scope<'env> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scope {{ ... }}")
    }
}

pub struct StatCountingScope<'scope, 'env: 'scope> {
    scope: &'scope Scope<'env>,
    stats: ScopeStats,
}

impl<'scope, 'env: 'scope> From<&'scope Scope<'env>> for StatCountingScope<'scope, 'env> {
    fn from(scope: &'scope Scope<'env>) -> Self {
        Self {
            scope,
            stats: Default::default(),
        }
    }
}

impl<'scope, 'env: 'scope> StatCountingScope<'scope, 'env> {
    /// Create a stat keeping scoped thread.
    ///
    /// This is essentially a wrapper function for `Scope::spawn`.
    /// To enable stat keeping for the scoped threads of a `Scope`, call the `Scope::stat_counting`
    /// function on a regular `Scope`.
    pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        self.scope.spawn(wrap_stat_counting(&self.stats, f))
    }

    /// Generates the base configuration for spawning a scoped thread, from which configuration
    /// methods can be chained.
    ///
    /// This is a wrapper function for `Scope::builder`.
    pub fn builder(&'scope self) -> StatCountingScopedThreadBuilder<'scope, 'env> {
        StatCountingScopedThreadBuilder {
            scope: self,
            builder: self.scope.builder(),
        }
    }

    /// Get the current count of running threads.
    ///
    /// This is not guaranteed to be completely accurate, as there is a small delay between the
    /// decrementing of the running count and the incrementing of the completed/panicked count.
    pub fn running_count(&self) -> usize {
        self.stats.running.load(Ordering::SeqCst)
    }

    /// Get the current count of completed threads.
    ///
    /// This is not guaranteed to be completely accurate, as there is a small delay between the
    /// decrementing of the running count and the incrementing of the completed/panicked count.
    pub fn completed_count(&self) -> usize {
        self.stats.completed.load(Ordering::SeqCst)
    }

    /// Get the current count of panicked threads.
    ///
    /// This is not guaranteed to be completely accurate, as there is a small delay between the
    /// decrementing of the running count and the incrementing of the completed/panicked count.
    pub fn panicked_count(&self) -> usize {
        self.stats.panicked.load(Ordering::SeqCst)
    }
}

impl<'scope, 'env: 'scope> fmt::Debug for StatCountingScope<'scope, 'env> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StatCountingScope {{ ... }}")
    }
}

/// Scoped thread configuration. Provides detailed control over the properties and behavior of new
/// scoped threads.
pub struct ScopedThreadBuilder<'scope, 'env: 'scope> {
    scope: &'scope Scope<'env>,
    builder: thread::Builder,
}

impl<'scope, 'env: 'scope> ScopedThreadBuilder<'scope, 'env> {
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
        F: FnOnce() -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        let result = Arc::new(Mutex::new(None));

        let join_handle = unsafe {
            let mut thread_result = Arc::clone(&result);
            builder_spawn_unchecked(self.builder, move || {
                *thread_result.lock().unwrap() = Some(f());
            })
        }?;

        let thread = join_handle.thread().clone();

        let join_state = JoinState::<T>::new(join_handle, result);
        let deferred_handle = Rc::new(RefCell::new(Some(join_state)));
        let my_handle = deferred_handle.clone();

        self.scope.joins.borrow_mut().push(Box::new(move || {
            let state = deferred_handle.borrow_mut().take();
            if let Some(state) = state {
                state.join().map(|_| ())
            } else {
                Ok(())
            }
        }));

        Ok(ScopedJoinHandle {
            inner: my_handle,
            thread: thread,
            _marker: PhantomData,
        })
    }
}

/// Scoped thread configuration with stat counting (a wrapper for `ScopedThreadBuilder`)
///
/// Provides detailed control over the properties and behavior of new scoped threads.
pub struct StatCountingScopedThreadBuilder<'scope, 'env: 'scope> {
    scope: &'scope StatCountingScope<'scope, 'env>,
    builder: ScopedThreadBuilder<'scope, 'env>,
}

impl<'scope, 'env: 'scope> StatCountingScopedThreadBuilder<'scope, 'env> {
    /// Wrapper function for `ScopedThreadBuilder::name`
    pub fn name(mut self, name: String) -> Self {
        self.builder = self.builder.name(name);
        self
    }

    /// Wrapper function for `ScopedThreadBuilder::stack_size`
    pub fn stack_size(mut self, size: usize) -> Self {
        self.builder = self.builder.stack_size(size);
        self
    }

    /// Spawns a new thread, and returns a join handle for it.
    pub fn spawn<F, T>(self, f: F) -> io::Result<ScopedJoinHandle<'scope, T>>
    where
        F: FnOnce() -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        self.builder.spawn(wrap_stat_counting(&self.scope.stats, f))
    }
}

//FIXME(oliver-giersch): Is this a good idea in combination with `Rc<RefCell<_>>`?
unsafe impl<'scope, T> Send for ScopedJoinHandle<'scope, T> {}
unsafe impl<'scope, T> Sync for ScopedJoinHandle<'scope, T> {}

/// A handle to a scoped thread
pub struct ScopedJoinHandle<'scope, T: 'scope> {
    // !Send + !Sync
    inner: Rc<RefCell<Option<JoinState<T>>>>,
    thread: thread::Thread,
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
        let state = (*self.inner.borrow_mut()).take();
        state.unwrap().join()
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

type ScopedThreadResult<T> = Arc<Mutex<Option<T>>>;

struct JoinState<T> {
    join_handle: thread::JoinHandle<()>,
    result: ScopedThreadResult<T>,
}

impl<T> JoinState<T> {
    fn new(join_handle: thread::JoinHandle<()>, result: ScopedThreadResult<T>) -> JoinState<T> {
        JoinState {
            join_handle,
            result,
        }
    }

    fn join(self) -> thread::Result<T> {
        let result = self.result;
        self.join_handle
            .join()
            .map(|_| result.lock().unwrap().take().unwrap())
    }
}

#[derive(Default)]
struct ScopeStats {
    running: AtomicUsize,
    completed: AtomicUsize,
    panicked: AtomicUsize,
}

/// Wrap a thread closure to enable stat counting with a `StatCountingScope`.
///
/// The lifetime transmutation is safe, because no references to the stats can leak out of the
/// closure and the guard is guaranteed to be dropped at the end of the closure.
fn wrap_stat_counting<'env, 'scope, F, T>(
    stats: &'scope ScopeStats,
    f: F,
) -> impl FnOnce() -> T + Send + 'env
where
    'env: 'scope,
    F: FnOnce() -> T,
    F: Send + 'env,
    T: Send + 'env,
{
    // This is necessary to satisfy the lifetime bound on the closure (`'env`) and to keep the
    // `StatKeepingScope` entirely separate/opt-in.
    let stats: &'env ScopeStats = unsafe { mem::transmute(stats) };
    move || {
        stats.running.fetch_add(1, Ordering::SeqCst);
        match panic::catch_unwind(panic::AssertUnwindSafe(|| f())) {
            Ok(res) => {
                stats.running.fetch_sub(1, Ordering::SeqCst);
                stats.completed.fetch_add(1, Ordering::SeqCst);
                return res;
            }
            Err(err) => {
                stats.running.fetch_sub(1, Ordering::SeqCst);
                stats.panicked.fetch_add(1, Ordering::SeqCst);
                panic::resume_unwind(err);
            }
        };
    }
}

#[doc(hidden)]
trait FnBox<T> {
    fn call_box(self: Box<Self>) -> T;
}

impl<T, F: FnOnce() -> T> FnBox<T> for F {
    fn call_box(self: Box<Self>) -> T {
        (*self)()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::{thread, time};

    const TIMES: usize = 10;
    const SMALL_STACK_SIZE: usize = 20;

    #[test]
    fn join() {
        let counter = AtomicUsize::new(0);
        scope(|scope| {
            let handle = scope.spawn(|| {
                counter.store(1, Ordering::Relaxed);
            });
            assert!(handle.join().is_ok());

            let panic_handle = scope.spawn(|| {
                panic!("\"My honey is running out!\", said Pooh.");
            });
            assert!(panic_handle.join().is_err());
        }).unwrap();

        // There should be sufficient synchronization.
        assert_eq!(1, counter.load(Ordering::Relaxed));
    }

    #[test]
    fn counter() {
        let counter = AtomicUsize::new(0);
        scope(|scope| {
            for _ in 0..TIMES {
                scope.spawn(|| {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        }).unwrap();

        assert_eq!(TIMES, counter.load(Ordering::Relaxed));
    }

    #[test]
    fn counter_builder() {
        let counter = AtomicUsize::new(0);
        scope(|scope| {
            for i in 0..TIMES {
                scope
                    .builder()
                    .name(format!("child-{}", i))
                    .stack_size(SMALL_STACK_SIZE)
                    .spawn(|| {
                        counter.fetch_add(1, Ordering::Relaxed);
                    })
                    .unwrap();
            }
        }).unwrap();

        assert_eq!(TIMES, counter.load(Ordering::Relaxed));
    }

    #[test]
    fn counter_panic() {
        let counter = AtomicUsize::new(0);
        let result = scope(|scope| {
            scope.spawn(|| {
                panic!("\"My honey is running out!\", said Pooh.");
            });
            thread::sleep(time::Duration::from_millis(100));

            for _ in 0..TIMES {
                scope.spawn(|| {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });

        assert_eq!(TIMES, counter.load(Ordering::Relaxed));
        assert!(result.is_err());
    }

    #[test]
    fn panic_twice() {
        let result = scope(|scope| {
            scope.spawn(|| {
                panic!();
            });
            panic!();
        });
        assert!(result.is_err());
    }
}
