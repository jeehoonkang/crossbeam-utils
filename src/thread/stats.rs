//! TODO: Module lvl docs

use std::io;
use std::fmt;
use std::mem;
use std::panic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::{Scope, ScopedThreadBuilder, ScopedJoinHandle};

impl<'env> Scope<'env> {
    /// Creates a scope that keeps track of the number of spawned, running
    /// completed and panicked threads.
    ///
    /// The stat counting scope cannot take ownership of the scope it is
    /// created from, so it would be possible to spawn threads from both the
    /// initial and the stat counting scope.
    /// To avoid this it is recommended to create the stat counting scope
    /// at the earliest opportunity and shadow the original scope variable
    /// with it.
    ///
    /// # Examples
    ///
    /// ```
    /// crossbeam_utils::thread::scope(|scope| {
    ///     let scope = scope.stat_counting();
    ///
    ///     let mut handles = Vec::with_capacity(3);
    ///     handles.push(scope.spawn(|| println!("thread #1")));
    ///     handles.push(scope.spawn(|| println!("thread #2")));
    ///     handles.push(scope.spawn(|| println!("thread #3")));
    ///
    ///     assert_eq!(3, scope.spawned_count());
    ///
    ///     for handle in handles {
    ///         handle.join().unwrap();
    ///     }
    ///
    ///     assert_eq!(3, scope.completed_count());
    /// });
    /// ```
    pub fn stat_counting<'scope>(&'scope self) -> StatCountingScope<'scope, 'env> {
        StatCountingScope::from(self)
    }
}

/// A wrapper for a scope that does stat counting on all newly spawned threads.
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
    /// This is essentially a wrapper function for [`Scope::spawn`].
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
    /// This is a wrapper function for [`Scope::builder`].
    pub fn builder(&'scope self) -> StatCountingScopedThreadBuilder<'scope, 'env> {
        StatCountingScopedThreadBuilder {
            scope: self,
            builder: self.scope.builder(),
        }
    }

    /// Get the current count of all spawned threads.
    pub fn spawned_count(&self) -> usize {
        self.stats.spawned.load(Ordering::SeqCst)
    }

    /// Get the current count of all running threads.
    pub fn running_count(&self) -> usize {
        let completed = self.stats.completed.load(Ordering::SeqCst);
        let panicked = self.stats.panicked.load(Ordering::SeqCst);
        let spawned = self.stats.spawned.load(Ordering::SeqCst);

        spawned - completed - panicked
    }

    /// Get the current count of all completed threads.
    pub fn completed_count(&self) -> usize {
        self.stats.completed.load(Ordering::SeqCst)
    }

    /// Get the current count of all panicked threads.
    pub fn panicked_count(&self) -> usize {
        self.stats.panicked.load(Ordering::SeqCst)
    }
}

impl<'scope, 'env: 'scope> fmt::Debug for StatCountingScope<'scope, 'env> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StatCountingScope {{ ... }}")
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

#[derive(Debug, Default)]
struct ScopeStats {
    spawned: AtomicUsize,
    completed: AtomicUsize,
    panicked: AtomicUsize,
}

/// Wraps a thread's main function, so that the referenced stats struct is kept
/// up to date.
fn wrap_stat_counting<'env, 'scope, F, T>(
    stats: &'scope ScopeStats,
    f: F
) -> impl FnOnce() -> T + Send + 'env
where
    'env: 'scope,
    F: FnOnce() -> T,
    F: Send + 'env,
    T: Send + 'env
{
    // This is necessary to satisfy the lifetime bound on the closure (`'env`) and to keep the
    // `StatKeepingScope` entirely separate/opt-in.
    // It is safe, however, because the ScopeStats can never leak from the scope.
    let stats: &'env ScopeStats = unsafe { mem::transmute(stats) };
    move || {
        stats.spawned.fetch_add(1, Ordering::SeqCst);
        match panic::catch_unwind(panic::AssertUnwindSafe(|| f())) {
            Ok(res) => {
                stats.completed.fetch_add(1, Ordering::SeqCst);
                return res;
            }
            Err(err) => {
                stats.panicked.fetch_add(1, Ordering::SeqCst);
                panic::resume_unwind(err);
            }
        };
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, Barrier};

    use super::super::*;
    use super::*;

    #[test]
    fn count_stats() {
        scope(|scope| {
            let scope = scope.stat_counting();
            let barrier = Arc::new(Barrier::new(3));

            let thread_barrier = Arc::clone(&barrier);
            let handle1 = scope.spawn(move || {
                thread_barrier.wait();
                thread_barrier.wait();
            });

            let thread_barrier = Arc::clone(&barrier);
            let handle2 = scope.spawn(move || {
                thread_barrier.wait();
                thread_barrier.wait();
            });

            barrier.wait();

            assert_eq!(2, scope.spawned_count());
            assert_eq!(2, scope.running_count());
            assert_eq!(0, scope.completed_count());
            assert_eq!(0, scope.panicked_count());

            barrier.wait();

            handle1.join().unwrap();
            handle2.join().unwrap();

            assert_eq!(2, scope.completed_count());
            assert_eq!(2, scope.panicked_count());
        });
    }

    #[test]
    fn count_panics() {
        const PANICS: [bool; 8] = [true, true, true, false, false, false, false, false];

        scope(|scope| {
            let scope = scope.stat_counting();
            let handles = (0..PANICS.len())
                .map(|id| scope.spawn(move || {
                    if PANICS[id] {
                        panic!();
                    }
                }))
                .collect::<Vec<_>>();

            for handle in handles {
                handle.join().unwrap();
            }

            assert_eq!(8, scope.spawned_count());
            assert_eq!(3, scope.panicked_count());
            assert_eq!(5, scope.completed_count());
            assert_eq!(0, scope.running_count());
        });
    }
}