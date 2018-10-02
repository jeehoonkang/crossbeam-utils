use std::fmt;
use std::process;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

/// Enables multiple threads to synchronize the beginning or end of some computation.
///
/// # Examples
///
/// ```
/// use crossbeam_utils::sync::WaitGroup;
/// use std::thread;
///
/// // Create a new wait group.
/// let wg = WaitGroup::new();
///
/// for _ in 0..4 {
///     // Create another reference to the wait group.
///     let wg = wg.clone();
///
///     thread::spawn(move || {
///         // Do some work.
///
///         // Drop the reference to the wait group.
///         drop(wg);
///     });
/// }
///
/// // Block until all threads have finished their work.
/// wg.wait();
/// ```
///
/// [`Barrier`]: https://doc.rust-lang.org/std/sync/struct.Barrier.html
pub struct WaitGroup {
    inner: Arc<Inner>,
}

/// Inner state of a `WaitGroup`.
struct Inner {
    cvar: Condvar,
    count: Mutex<usize>,
}

impl WaitGroup {
    /// Creates a new wait group and returns the single reference to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::sync::WaitGroup;
    ///
    /// let wg = WaitGroup::new();
    /// ```
    pub fn new() -> WaitGroup {
        WaitGroup {
            inner: Arc::new(Inner {
                cvar: Condvar::new(),
                count: Mutex::new(1),
            }),
        }
    }

    /// Drops this reference and waits until all other references are dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_utils::sync::WaitGroup;
    /// use std::thread;
    ///
    /// let wg = WaitGroup::new();
    ///
    /// thread::spawn({
    ///     let wg = wg.clone();
    ///     move || {
    ///         // Block until both threads have reached `wait()`.
    ///         wg.wait();
    ///     }
    /// });
    ///
    /// // Block until both threads have reached `wait()`.
    /// wg.wait();
    /// ```
    pub fn wait(self) {
        if *self.inner.count.lock() == 1 {
            return;
        }

        let inner = self.inner.clone();
        drop(self);

        let mut count = inner.count.lock();
        while *count > 0 {
            inner.cvar.wait(&mut count);
        }
    }
}

impl Drop for WaitGroup {
    fn drop(&mut self) {
        *self.inner.count.lock() -= 1;
        self.inner.cvar.notify_all();
    }
}

impl Clone for WaitGroup {
    fn clone(&self) -> WaitGroup {
        let mut count = self.inner.count.lock();
        *count += 1;

        if *count > isize::max_value() as usize {
            process::abort();
        }

        WaitGroup {
            inner: self.inner.clone(),
        }
    }
}

impl fmt::Debug for WaitGroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let count = self.inner.count.lock();
        write!(f, "WaitGroup {{ count: {:?} }}", *count)
    }
}
