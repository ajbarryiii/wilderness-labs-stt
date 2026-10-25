// u32io: little-endian u32 arrays between files and Array<U32>, with no byte list in between
// (Base's File.read_bytes makes one List cell per byte). The compiled program keeps an Array<U32>
// as one BUF block of 32-bit cells (a block of class c holds 2^c cells, cell i at byte 4i), so on a
// little-endian host a file's bytes are the cells' bits: pread and write move them directly. The
// block does not move while the helper thread touches it, and the array is affine, so no one else
// reads or writes it until the effect answers. Modelled on the hub package bend-ml-binary-read
// (read_f32.c); written for this planner, with the same shape.
//
// U32s.read(file, offset, n, a): n cells from byte offset into cells 0 .. n-1 of a. Answers the
//   file, the array and the cells read, or the error; a short read fails with EIO.
// U32s.write(file, n, a): cells 0 .. n-1 of a, written at the file's position.
// Both fail with EINVAL, touching nothing, when n exceeds a's capacity.

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "u32io moves the file's bytes as they are: it needs a little-endian host"
#endif

// w->size holds the bytes to move; w->made the read offset; w->data points into the array's block
static void u32io_read_call(IoWork* w) {
  u64 want = w->size;
  u64 got  = 0;
  while (got < want) {
    u64     step = want - got < (1ull << 30) ? want - got : (1ull << 30);
    ssize_t n    = pread((int)w->hand, w->data + got, step, (off_t)w->made + (off_t)got);
    if (n <= 0) {
      if (n == 0) {
        w->code = EIO;
      } else {
        io_sys_end(w, n);
      }
      break;
    }
    got += (u64)n;
  }
  w->size = got;
}

static void u32io_write_call(IoWork* w) {
  u64 want = w->size;
  u64 got  = 0;
  while (got < want) {
    u64     step = want - got < (1ull << 30) ? want - got : (1ull << 30);
    ssize_t n    = write((int)w->hand, w->data + got, step);
    if (n <= 0) {
      // no progress would retry forever (as Base's file_write.c)
      if (n == 0) {
        w->code = EIO;
      } else {
        io_sys_end(w, n);
      }
      break;
    }
    got += (u64)n;
  }
  w->size = got;
}

// nothing to free: w->data is the array's own block
static Term u32io_read_pack(Env e, IoWork* w) {
  Term r = io_res(e, w, (Term)(u32)(w->size / 4));
  return io_tup(e, io_hand(w->hand), io_tup(e, w->item, r));
}

static Term u32io_write_pack(Env e, IoWork* w) {
  Term r = io_res(e, w, term_pak(CID(Unit), 0));
  return io_tup(e, io_hand(w->hand), io_tup(e, w->item, r));
}

// Checks a and n and points w at the block; answers the failure Term, or 0 when the work may start.
static Term u32io_start(Env e, Term file, Term a, u64 n, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(file);
  w->item = a;
  w->code = 0;
  if (term_tag(a) != TAG_BUF || n > (1ull << blk_cls(a))) {
    Term r = io_fail(e, EINVAL, NULL);
    return io_tup(e, io_hand(w->hand), io_tup(e, a, r));
  }
  w->size = n * 4;
  w->data = (char*)blk_ptr(e.mem, blk_loc(e.mem, a), 0);
  return 0;
}

#ifdef CID(U32s.read)

Term u32io_read_run(Env e, Term* f, IoWork* w) {
  Term bad = u32io_start(e, f[0], f[3], (u64)(u32)f[2], w);
  if (bad != 0) {
    return bad;
  }
  w->made = (intptr_t)(u32)f[1];
  return io_work(w, u32io_read_call, u32io_read_pack);
}

static void __attribute__((constructor)) u32io_read_use(void) {
  io_eff(CID(U32s.read), u32io_read_run);
}

#endif

#ifdef CID(U32s.write)

Term u32io_write_run(Env e, Term* f, IoWork* w) {
  Term bad = u32io_start(e, f[0], f[2], (u64)(u32)f[1], w);
  if (bad != 0) {
    return bad;
  }
  return io_work(w, u32io_write_call, u32io_write_pack);
}

static void __attribute__((constructor)) u32io_write_use(void) {
  io_eff(CID(U32s.write), u32io_write_run);
}

#endif
