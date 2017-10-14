use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// 64 bytes is the most common cache line size on modern machines.
const CACHE_LINE_BYTES: usize = 64;

#[cfg(not(feature = "nightly"))]
struct Inner<T> {
    bytes: [u8; CACHE_LINE_BYTES],
    _marker: PhantomData<T>,
}

#[cfg(feature = "nightly")]
#[repr(align(64))]
#[allow(unions_with_drop_fields)]
union Inner<T> {
    bytes: [u8; CACHE_LINE_BYTES],
    _value: T,
}

/// Pads `T` to the length of a cache line.
///
/// Sometimes concurrent programming requires a piece of data to be padded out to the size of a
/// cacheline to avoid "false sharing": cache lines being invalidated due to unrelated concurrent
/// activity. Use this type when you want to *avoid* cache locality.
///
/// At the moment, cache lines are assumed to be 64 bytes on all architectures.
/// Note that, while the size of `CachePadded<T>` is 64 bytes, the alignment may not be.
#[allow(unions_with_drop_fields)]
pub struct CachePadded<T> {
    inner: Inner<T>,

    /// `[T; 0]` ensures correct alignment.
    /// `PhantomData<T>` signals that `CachePadded<T>` contains a `T`.
    _marker: ([T; 0], PhantomData<T>),
}

unsafe impl<T: Send> Send for CachePadded<T> {}
unsafe impl<T: Sync> Sync for CachePadded<T> {}

impl<T> CachePadded<T> {
    /// Pads a value to the length of a cache line.
    pub fn new(t: T) -> CachePadded<T> {
        assert!(mem::size_of::<T>() <= mem::size_of::<CachePadded<T>>());
        assert!(mem::align_of::<T>() <= mem::align_of::<CachePadded<T>>());

        unsafe {
            let mut padded = CachePadded {
                inner: mem::uninitialized(),
                _marker: ([], PhantomData),
            };
            let p: *mut T = &mut *padded;
            ptr::write(p, t);
            padded
        }
    }
}

impl<T> Drop for CachePadded<T> {
    fn drop(&mut self) {
        let p: *mut T = &mut **self;
        unsafe {
            ptr::drop_in_place(p);
        }
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*(&self.inner.bytes as *const _ as *const T) }
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *(&mut self.inner.bytes as *mut _ as *mut T) }
    }
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T: Clone> Clone for CachePadded<T> {
    fn clone(&self) -> Self {
        unsafe {
            let mut new: Self = CachePadded {
                inner: mem::uninitialized(),
                _marker: ([], PhantomData),
            };

            let src: *const T = self.deref();
            let dest: *mut T = new.deref_mut();
            ptr::copy_nonoverlapping(src, dest, 1);

            new
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for CachePadded<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner: &T = &*self;
        write!(f, "CachePadded {{ {:?} }}", inner)
    }
}

impl<T> From<T> for CachePadded<T> {
    fn from(t: T) -> Self {
        CachePadded::new(t)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn store_u64() {
        let x: CachePadded<u64> = CachePadded::new(17);
        assert_eq!(*x, 17);
    }

    #[test]
    fn store_pair() {
        let x: CachePadded<(u64, u64)> = CachePadded::new((17, 37));
        assert_eq!(x.0, 17);
        assert_eq!(x.1, 37);
    }

    #[test]
    fn distance() {
        let arr = [CachePadded::new(17u8), CachePadded::new(37u8)];
        let a = &*arr[0] as *const u8;
        let b = &*arr[1] as *const u8;
        assert_eq!(a.wrapping_offset(64), b);
    }

    #[test]
    fn different_sizes() {
        CachePadded::new(17u8);
        CachePadded::new(17u16);
        CachePadded::new(17u32);
        CachePadded::new([17u64; 0]);
        CachePadded::new([17u64; 1]);
        CachePadded::new([17u64; 2]);
        CachePadded::new([17u64; 3]);
        CachePadded::new([17u64; 4]);
        CachePadded::new([17u64; 5]);
        CachePadded::new([17u64; 6]);
        CachePadded::new([17u64; 7]);
        CachePadded::new([17u64; 8]);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    #[should_panic]
    fn large() {
        CachePadded::new([17u64; 9]);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn large() {
        let a = [17u64; 9];
        let b = CachePadded::new(a);
        assert!(mem::size_of_val(&a) <= mem::size_of_val(&b));
    }

    #[test]
    fn debug() {
        assert_eq!(format!("{:?}", CachePadded::new(17u64)), "CachePadded { 17 }");
    }

    #[test]
    fn drops() {
        let count = Cell::new(0);

        struct Foo<'a>(&'a Cell<usize>);

        impl<'a> Drop for Foo<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let a = CachePadded::new(Foo(&count));
        let b = CachePadded::new(Foo(&count));

        assert_eq!(count.get(), 0);
        drop(a);
        assert_eq!(count.get(), 1);
        drop(b);
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn clone() {
        let a = CachePadded::new(17);
        let b = a.clone();
        assert_eq!(*a, *b);
    }
}
