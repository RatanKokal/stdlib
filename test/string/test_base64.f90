! SPDX-Identifier: MIT
module test_base64
    use testdrive, only : new_unittest, unittest_type, error_type, check
    use stdlib_base64, only : base64_encode, base64_decode, base64_encode_into, base64_decode_into
    use stdlib_kinds, only : int8, int32, dp, lk
    implicit none

contains

    subroutine collect_base64(testsuite)
        type(unittest_type), allocatable, intent(out) :: testsuite(:)

        testsuite = [ &
            new_unittest("base64-known-vectors", test_known_vectors), &
            new_unittest("base64-decode-whitespace", test_decode_whitespace), &
            new_unittest("base64-decode-invalid", test_decode_invalid), &
            new_unittest("base64-roundtrip-int32", test_roundtrip_int32), &
            new_unittest("base64-roundtrip-real", test_roundtrip_real), &
            new_unittest("base64-roundtrip-complex", test_roundtrip_complex), &
            new_unittest("base64-roundtrip-logical", test_roundtrip_logical), &
            new_unittest("base64-rank0", test_rank0_encode), &
            new_unittest("base64-subroutine-api", test_subroutine_api), &
            new_unittest("base64-avx2-fat-loops", test_avx2_fat_loops), &
            new_unittest("base64-fused-despacer", test_fused_despacer), &
            new_unittest("base64-unaligned-memory", test_unaligned_memory), &
            new_unittest("base64-strict-compliance", test_strict_compliance) &
            ]
    end subroutine collect_base64

    ! =========================================================================
    ! STANDARD COMPLIANCE TESTS
    ! =========================================================================

    subroutine test_known_vectors(error)
        type(error_type), allocatable, intent(out) :: error

        call check(error, base64_encode([int(77, int8), int(97, int8), int(110, int8)]) == "TWFu")
        if (allocated(error)) return
        call check(error, base64_encode([int(77, int8), int(97, int8)]) == "TWE=")
        if (allocated(error)) return
        call check(error, base64_encode([int(77, int8)]) == "TQ==")
        if (allocated(error)) return

        call check(error, base64_decode("TWFu") == "Man")
        if (allocated(error)) return
        call check(error, base64_decode("TWE=") == "Ma")
        if (allocated(error)) return
        call check(error, base64_decode("TQ==") == "M")
    end subroutine test_known_vectors

    subroutine test_decode_whitespace(error)
        type(error_type), allocatable, intent(out) :: error
        call check(error, base64_decode("T W" // new_line("a") // "Fu") == "Man")
    end subroutine test_decode_whitespace

    subroutine test_decode_invalid(error)
        type(error_type), allocatable, intent(out) :: error
        call check(error, base64_decode("abc") == "")
        if (allocated(error)) return
        call check(error, base64_decode("A===") == "")
        if (allocated(error)) return
        call check(error, base64_decode("AA=A") == "")
    end subroutine test_decode_invalid

    subroutine test_roundtrip_int32(error)
        type(error_type), allocatable, intent(out) :: error
        integer(int32) :: vals(4), got(4)
        character(len=:), allocatable :: enc, dec

        vals = [1_int32, -2_int32, 1024_int32, -4096_int32]

        enc = base64_encode(vals)
        dec = base64_decode(enc)
        got = transfer(dec, got)

        call check(error, all(got == vals))
    end subroutine test_roundtrip_int32

    subroutine test_roundtrip_real(error)
        type(error_type), allocatable, intent(out) :: error
        real(dp) :: vals(4), got(4)
        character(len=:), allocatable :: enc, dec

        vals = [1.5_dp, -2.25_dp, 0.125_dp, 9.0_dp]

        enc = base64_encode(vals)
        dec = base64_decode(enc)
        got = transfer(dec, got)

        call check(error, all(got == vals))
    end subroutine test_roundtrip_real

    subroutine test_roundtrip_complex(error)
        type(error_type), allocatable, intent(out) :: error
        complex(dp) :: vals(3), got(3)
        character(len=:), allocatable :: enc, dec

        vals = [cmplx(1.0_dp, 2.0_dp, dp), cmplx(-3.0_dp, 0.5_dp, dp), cmplx(0.0_dp, -4.0_dp, dp)]

        enc = base64_encode(vals)
        dec = base64_decode(enc)
        got = transfer(dec, got)

        call check(error, all(got == vals))
    end subroutine test_roundtrip_complex

    subroutine test_roundtrip_logical(error)
        type(error_type), allocatable, intent(out) :: error
        logical(lk) :: vals(5), got(5)
        character(len=:), allocatable :: enc, dec

        vals = [.true._lk, .false._lk, .true._lk, .true._lk, .false._lk]

        enc = base64_encode(vals)
        dec = base64_decode(enc)
        
        got = transfer(dec, got)

        call check(error, all((vals .neqv. .false._lk) .eqv. (got .neqv. .false._lk)))
    end subroutine test_roundtrip_logical

    subroutine test_rank0_encode(error)
        type(error_type), allocatable, intent(out) :: error
        integer(int32) :: v

        v = 42_int32
        call check(error, len(base64_encode(v)) > 0)
    end subroutine test_rank0_encode

    subroutine test_subroutine_api(error)
        type(error_type), allocatable, intent(out) :: error
        integer(int8) :: data(3)
        character(len=4) :: enc_out
        character(len=3) :: dec_out
        integer :: decoded_len
        logical :: err_flag

        data = [int(77, int8), int(97, int8), int(110, int8)] ! "Man"

        ! 1. Test Encode Subroutine
        call base64_encode_into(data, enc_out)
        call check(error, enc_out == "TWFu")
        if (allocated(error)) return

        ! 2. Test Decode Subroutine
        call base64_decode_into(enc_out, dec_out, decoded_len, err_flag)
        call check(error, .not. err_flag)
        if (allocated(error)) return
        call check(error, decoded_len == 3)
        if (allocated(error)) return
        call check(error, dec_out(1:decoded_len) == "Man")
    end subroutine test_subroutine_api

    ! =========================================================================
    ! GOLD STANDARD HARDWARE & EDGE CASE TESTS
    ! =========================================================================

    subroutine test_avx2_fat_loops(error)
        ! Tests payloads large enough to saturate the 192-byte encoder and 128-byte decoder loops.
        type(error_type), allocatable, intent(out) :: error
        integer, parameter :: N = 10000 ! 10 KB payload
        integer(int8), allocatable :: raw(:)
        character(len=:), allocatable :: enc, dec
        integer :: i
        
        allocate(raw(N))
        do i = 1, N
            raw(i) = int(mod(i, 256) - 128, int8) ! Cover full binary spectrum
        end do

        enc = base64_encode(raw)
        ! 10000 bytes = 13336 base64 chars. Easily clears the 200/160 byte thresholds.
        call check(error, len(enc) == 13336)
        if (allocated(error)) return

        dec = base64_decode(enc)
        call check(error, len(dec) == N)
        if (allocated(error)) return
        
        call check(error, all(transfer(dec, raw) == raw))
    end subroutine test_avx2_fat_loops

    subroutine test_fused_despacer(error)
        ! Tests the L1 Queue by aggressively inserting spaces and newlines
        type(error_type), allocatable, intent(out) :: error
        integer, parameter :: N = 2000 
        integer(int8), allocatable :: raw(:)
        character(len=:), allocatable :: enc, mime_enc, dec
        integer :: i, j, mime_len
        
        allocate(raw(N))
        do i = 1, N; raw(i) = int(mod(i, 26) + 65, int8); end do
        
        ! Clean encoding
        enc = base64_encode(raw)
        
        ! Build a MIME string (newline every 76 characters)
        mime_len = len(enc) + (len(enc) / 76) * 2 + 10 ! extra padding for random spaces
        allocate(character(len=len(enc) * 2) :: mime_enc)
        
        j = 1
        do i = 1, len(enc)
            mime_enc(j:j) = enc(i:i)
            j = j + 1
            if (mod(i, 76) == 0) then
                mime_enc(j:j+1) = char(13) // char(10) ! \r\n
                j = j + 2
            end if
            ! Throw in random spaces to totally disrupt the 32-byte boundaries
            if (mod(i, 113) == 0) then
                mime_enc(j:j) = " "
                j = j + 1
            end if
        end do
        
        ! Decode the heavily fragmented MIME string
        dec = base64_decode(mime_enc(1:j-1))
        
        call check(error, len(dec) == N)
        if (allocated(error)) return
        call check(error, all(transfer(dec, raw) == raw))
    end subroutine test_fused_despacer

    subroutine test_unaligned_memory(error)
        ! Slicing arrays forces Fortran to pass unaligned memory addresses to C.
        ! This guarantees _mm256_storeu_si256 is used properly without segfaulting.
        type(error_type), allocatable, intent(out) :: error
        integer, parameter :: N = 1000
        integer(int8), allocatable :: raw(:), got(:)
        character(len=:), allocatable :: enc
        character(len=2000) :: dec_buffer
        integer :: decoded_len, i
        logical :: err_flag
        
        allocate(raw(N))
        do i = 1, N; raw(i) = 65_int8; end do ! fill with 'A'
        
        ! Slice starting at index 3 (offset of 2 bytes, destroying standard 32-byte alignment)
        enc = base64_encode(raw(3:N-2))
        
        ! Decode into a sliced string buffer (forces unaligned destination)
        call base64_decode_into(enc, dec_buffer(5:1800), decoded_len, err_flag)
        
        call check(error, .not. err_flag)
        if (allocated(error)) return
        call check(error, decoded_len == (N - 4))
        if (allocated(error)) return
        
        allocate(got(decoded_len))
        got = transfer(dec_buffer(5:5+decoded_len-1), got)
        call check(error, all(got == 65_int8))
    end subroutine test_unaligned_memory

    subroutine test_strict_compliance(error)
        ! Tests the specific RFC 4648 vulnerabilities patched in the scalar tail
        type(error_type), allocatable, intent(out) :: error
        character(len=10) :: dec_out
        integer :: decoded_len
        logical :: err_flag

        ! 1. The High-Bit Injection Trap
        ! Character 193 (0xC1) masks to 65 ('A') if & 0x7F is used blindly.
        call base64_decode_into("TW" // char(193) // "u", dec_out, decoded_len, err_flag)
        call check(error, err_flag .eqv. .true.)
        if (allocated(error)) return

        ! 2. The Truncation Trap
        ! "TWFu" is valid. "TWF" is length 3, which is mathematically impossible for Base64.
        call base64_decode_into("TWF", dec_out, decoded_len, err_flag)
        call check(error, err_flag .eqv. .true.)
        if (allocated(error)) return

        ! 3. Invalid Padding Trap
        ! "TWE=" is valid ("Ma"). "TWE*" is not.
        call base64_decode_into("TWE*", dec_out, decoded_len, err_flag)
        call check(error, err_flag .eqv. .true.)
        if (allocated(error)) return
        
        ! 4. Buffer too small Trap
        ! "SGVsbG8gV29ybGQ=" decodes to "Hello World" (11 bytes). 
        ! We only provide a 10-byte buffer.
        call base64_decode_into("SGVsbG8gV29ybGQ=", dec_out, decoded_len, err_flag)
        call check(error, err_flag .eqv. .true.)
        if (allocated(error)) return
        call check(error, decoded_len == 0)

    end subroutine test_strict_compliance

end module test_base64


! =========================================================================
! MAIN TEST RUNNER
! =========================================================================
program tester
    use, intrinsic :: iso_fortran_env, only : error_unit
    use testdrive, only : run_testsuite, new_testsuite, testsuite_type
    use test_base64, only : collect_base64
    implicit none
    
    integer :: stat, is
    type(testsuite_type), allocatable :: testsuites(:)
    character(len=*), parameter :: fmt = '("#", *(1x, a))'

    stat = 0

    testsuites = [ &
        new_testsuite("base64", collect_base64) &
        ]

    do is = 1, size(testsuites)
        write(error_unit, fmt) "Testing:", testsuites(is)%name
        call run_testsuite(testsuites(is)%collect, error_unit, stat)
    end do

    if (allocated(testsuites)) deallocate(testsuites)

    if (stat > 0) then
        write(error_unit, '(i0, 1x, a)') stat, "test(s) failed!"
        error stop
    end if
end program tester