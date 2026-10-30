##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information
##

##  1D batch of 1D DFTs: R2C & C2R

##  Parameters expected to be defined ahead of this code:
##  fftlen -- length of FFT
##  nbatch -- batch size
##  rdstride -- read stride type { APar | AVec }
##  wrstride -- write stride type { APar | AVec }
##  codefor -- which architecture to generate code for { CUDA | CPU | HIP }
##  fwd -- Transform direction { true | false }
##  libdir -- name of directory in which to write files
##  file_suffix -- suffix part of output file name

Load(fftx);
ImportAll(fftx);
Import(realdft);

##  If the variable createJIT is defined and set true then load the jit module
if ( IsBound(createJIT) and createJIT ) then
    ImportAll(simt);
    Load(jit);
    Import(jit);
fi;

if codefor = "CUDA" then
    conf := LocalConfig.fftx.confGPU();
elif codefor = "HIP" then
    conf := FFTXGlobals.defaultHIPConf();
elif codefor = "SYCL" then
    conf := FFTXGlobals.defaultOpenCLConf();
elif codefor = "CPU" then
    conf := LocalConfig.fftx.defaultConf();
fi;

if fwd then
    prefix := "fftx_prdftbat_";
    jitpref := "cache_prdftbat_";
    sign   := -1;
else
    prefix := "fftx_iprdftbat_";
    jitpref := "cache_iprdftbat_";
    sign   := 1;
fi;

##  vmecpp customization: post-transform per-bin diagonal scaling.
##  PRDFT(fftlen) outputs nhalf = fftlen/2+1 complex bins (= 2*nhalf reals).
##  vmecpp's accumulate loop multiplies each bin by nscale[n] (= 1 for n=0,
##  sqrt(2) for n>=1).  Folding this into the kernel via RCDiag of length
##  nhalf (real scalar per complex bin -- both re and im get scaled by the
##  same factor) lets SPIRAL keep it as a tagged complex operator and avoids
##  unrolling 2*nhalf*nbatch constants.
nhalf_v := Int(fftlen / 2) + 1;
nscale_d := List([0..nhalf_v-1], k -> When(k = 0, 1.0, Sqrt(2.0)));
##  RCDiag with realified twiddles (re,im per complex bin).  We want a real
##  scalar nscale[k] applied to bin k -- that's complex multiply by nscale[k]+0i,
##  i.e. interleaved [s_0, 0, s_1, 0, s_2, 0, ...].
diag_per_item := Concatenation(List(nscale_d, x -> [x, 0.0]));
postDiag := RCDiag(FList(TReal, diag_per_item));

if 1 = 1 then
    name := prefix::StringInt(fftlen)::"_bat_"::StringInt(nbatch)::"_"::wrstride::"_"::rdstride::"_"::codefor;
    jitname := jitpref::StringInt(fftlen)::"_bat_"::StringInt(nbatch)::"_"::wrstride::"_"::rdstride::"_"::codefor::".txt";

    PrintLine("fftx_prdftbat: name = ", name, " length = ", fftlen, " bat = ", nbatch, " write stride: ", wrstride,
              " read stride: ", rdstride );

    _wr := APar; if wrstride = "AVec" then _wr := AVec; fi;
    _rd := APar; if rdstride = "AVec" then _rd := AVec; fi;

    ##  we have 8 cases (fwd/inverse and APar/AVec combos) all possibly different:
    if fwd then
        if rdstride = "APar" and wrstride = "APar" then
            t := let ( TFCall ( TTensorI ( postDiag * PRDFT ( fftlen, sign ), nbatch, APar, _rd ),
                                rec ( fname := name, params := [] ) ) );
        elif rdstride = "APar" and wrstride = "AVec" then
            t := let ( TFCall ( Prm ( fTensor ( L ( PRDFT1 ( fftlen, sign ).dims()[1]/2 * nbatch,
                                                    PRDFT1 ( fftlen, sign ).dims()[1]/2 ),
                                                fId (2) ) ) *
                                TTensorI ( postDiag * PRDFT1 ( fftlen, sign ), nbatch, APar, _rd ),
                                rec ( fname := name, params := [] ) ) );
        elif rdstride = "AVec" and wrstride = "APar" then
            t := let ( TFCall ( TTensorI ( postDiag * PRDFT ( fftlen, sign ), nbatch, APar, _rd ),
                                rec ( fname := name, params := [] ) ) );
        else            ##  rdstride = "AVec" and wrstride = "AVec"
            t := let ( TFCall ( Prm ( fTensor ( L ( PRDFT1 ( fftlen, sign).dims()[1]/2 * nbatch,
                                                    PRDFT1 ( fftlen, sign).dims()[1]/2),
                                                fId (2) ) ) *
                                TTensorI ( postDiag * PRDFT1 ( fftlen, sign ), nbatch, APar, _rd ),
                                rec ( fname := name, params := [] ) ) );
        fi;
    else
        if rdstride = "APar" and wrstride = "APar" then
            t := let ( TFCall ( TTensorI ( IPRDFT ( fftlen, sign ), nbatch, _wr, _rd ),
                                rec ( fname := name, params := [] ) ) );
        elif rdstride = "APar" and wrstride = "AVec" then
            t := let ( TFCall ( TTensorI ( IPRDFT ( fftlen, sign ), nbatch, _wr, _rd ),
                                rec ( fname := name, params := [] ) ) );
        elif rdstride = "AVec" and wrstride = "APar" then
            t := let ( TFCall ( TTensorI ( IPRDFT ( fftlen, sign ), nbatch, _wr, APar ) *
                                Prm ( fTensor ( L ( IPRDFT1 ( fftlen, sign ).dims()[2]/2 * nbatch, nbatch ), fId (2) ) ),
                                rec ( fname := name, params := [] ) ) );
        else            ##  rdstride = "AVec" and wrstride = "AVec"
            t := let ( TFCall ( TTensorI ( IPRDFT ( fftlen, sign ), nbatch, _wr, APar ) *
                                Prm ( fTensor ( L ( IPRDFT1 ( fftlen, sign ).dims()[2]/2 * nbatch, nbatch ), fId (2) ) ),
                                rec ( fname := name, params := [] ) ) );
        fi;
    fi;

    opts := conf.getOpts(t);
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;
    Add ( opts.includes, "<float.h>" );
    tt := opts.tagIt(t);

    _tt := opts.preProcess(tt);
    rt := opts.search(_tt);

    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo ( libdir::"/"::name::file_suffix, opts.prettyPrint(c) );
fi;

##  If the variable createJIT is defined and set true then output the JIT code to a file
if ( IsBound(createJIT) and createJIT ) then
    cachedir := GetEnv("FFTX_HOME");
    if (cachedir = "") then cachedir := "../.."; fi;
    cachedir := cachedir::"/cache_jit_files/";
    if ( codefor = "HIP" ) then PrintTo ( cachedir::jitname, PrintHIPJIT ( c, opts ) ); fi;
    if ( codefor = "CUDA" ) then PrintTo ( cachedir::jitname, PrintJIT2 ( c, opts ) ); fi;
    if ( codefor = "SYCL" ) then PrintTo ( cachedir::jitname, PrintOpenCLJIT ( c, opts ) ); fi;
    if ( codefor = "CPU" ) then PrintTo ( cachedir::jitname, opts.prettyPrint ( c ) ); fi;
fi;

if ( codefor = "CPU" ) then Exit(0); fi;
