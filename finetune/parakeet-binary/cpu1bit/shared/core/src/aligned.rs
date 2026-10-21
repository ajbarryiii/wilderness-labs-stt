//! 128-byte aligned (the M1's cache-line size, `sysctl hw.cachelinesize` = 128), zero-initialised
//! buffers of plain-old-data values.

use std::alloc::{Layout, alloc_zeroed, dealloc};

pub const ALIGN: usize = 128;

/// Types for which the all-zero bit pattern is a valid value.
pub unsafe trait Zeroable: Copy {}
unsafe impl Zeroable for u8 {}
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for u16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u32 {}
unsafe impl Zeroable for u64 {}
unsafe impl Zeroable for i64 {}
unsafe impl Zeroable for f32 {}

pub struct AlignedVec<T: Zeroable> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Zeroable + Send> Send for AlignedVec<T> {}
unsafe impl<T: Zeroable + Sync> Sync for AlignedVec<T> {}

impl<T: Zeroable> AlignedVec<T> {
    pub fn zeroed(len: usize) -> Self {
        if len == 0 || size_of::<T>() == 0 {
            return AlignedVec { ptr: std::ptr::NonNull::dangling().as_ptr(), len };
        }
        let layout = Self::layout(len);
        let ptr = unsafe { alloc_zeroed(layout) } as *mut T;
        assert!(!ptr.is_null(), "allocation of {} bytes failed", layout.size());
        AlignedVec { ptr, len }
    }

    fn layout(len: usize) -> Layout {
        let bytes = len.checked_mul(size_of::<T>()).expect("size overflow");
        Layout::from_size_align(bytes.div_ceil(ALIGN) * ALIGN, ALIGN).unwrap()
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Heap bytes actually allocated (the layout rounds up to the alignment); 0 when nothing is allocated.
    pub fn heap_bytes(&self) -> usize {
        if self.len == 0 || size_of::<T>() == 0 { 0 } else { Self::layout(self.len).size() }
    }
    pub fn bytes(&self) -> usize {
        self.len * size_of::<T>()
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Zeroable> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.len != 0 && size_of::<T>() != 0 {
            unsafe { dealloc(self.ptr as *mut u8, Self::layout(self.len)) };
        }
    }
}

impl<T: Zeroable> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Zeroable> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
