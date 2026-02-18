# PySCRDT

A Python module for calculating **resonance driving terms (RDTs)** and **non-linear
detuning coefficients** arising from the transverse space charge potential in a
circular accelerator.

Twiss data can be supplied either from a **MAD-X** Twiss table or from an
**X-suite** tracker dictionary.

---

## Physics background

Space charge (SC) is the self-field of a bunched beam. In a circular machine the
transverse SC potential can be expanded in action-angle coordinates; each term
in this expansion drives a resonance of order `m·Qx + n·Qy = l` (where `Qx`,
`Qy` are the horizontal and vertical tunes and `l` is the harmonic number).
PySCRDT evaluates those driving terms by:

1. Expanding the SC potential analytically with SymPy to the requested order.
2. Integrating over the Gaussian transverse distribution (beam size at every point
   in the lattice).
3. Accumulating the phasor integral around the ring using the interpolated Twiss
   functions.

The result is a complex number whose modulus is the RDT amplitude and whose
argument is the RDT phase. The same machinery is used to compute amplitude-
dependent tune shifts (detuning).

A detailed derivation is given in the accompanying note:
<https://cds.cern.ch/record/2696190/files/CERN-ACC-NOTE-2019-0046.pdf>

For a companion tool that estimates the SC tune spread analytically see:
<https://github.com/fasvesta/tune-spread>

---

## Installation

```bash
# from PyPI / GitHub (editable dev install)
pip install git+https://github.com/fasvesta/PySCRDT.git

# local editable install (after cloning)
pip install -e ".[dev]"
```

Requires Python ≥ 3.9, NumPy ≥ 1.20, SymPy ≥ 1.8.

---

## Quick-start

### Mode 3 — resonance described by (m, n, l)

This is the standard mode: a resonance is identified by its orders `m` (horizontal),
`n` (vertical), and harmonic `l` such that `m·Qx + n·Qy = l`.

```python
from PySCRDT import PySCRDT

sc = PySCRDT()

# 1. Beam parameters (LHC-like proton beam as an example)
sc.setParameters(
    intensity   = 1.6e11,   # bunch population [protons]
    bunchLength = 0.09,     # RMS bunch length [m]
    ro          = 1.5347e-18,  # classical proton radius [m]
    emittance_x = 2.5e-6,  # normalised horizontal emittance [m·rad]
    emittance_y = 2.5e-6,  # normalised vertical emittance [m·rad]
    dpp_rms     = 1.1e-4,  # RMS Δp/p
    dpp         = 0.0,     # single-particle Δp/p (non-zero → feed-down)
    harmonic    = 1,       # number of bunches / RF harmonic
)

# 2. Load lattice Twiss data (MAD-X output file)
sc.prepareData("twiss.tfs")

# 3. Select resonance: 4th-order horizontal (4·Qx = 28)
sc.setMode(3)
sc.setOrder([4, 0, 28])  # [m, n, l]

# 4. Compute and retrieve the RDT
result = sc.getResonanceDrivingTerms()
print(f"RDT amplitude : {result['Amplitude']:.4e}")
print(f"RDT phase     : {result['Phase']:.4f} rad")
print(f"RDT (complex) : {result['RDT']}")
```

To scan over all harmonics (i.e. the resonance lattice sum regardless of `l`)
pass `'any'` as the third element:

```python
sc.setOrder([4, 0, "any"])
```

---

### Mode 5 — resonance described by (h, i, j, k, l)

Mode 5 separates the horizontal order into `(h, i)` and the vertical order into
`(j, k)`, giving more control over the resonance topology. The driving term
amplitude is governed by `h+i` (H) and `j+k` (V) while the phase accumulation
is governed by `h-i` (H) and `j-k` (V).

```python
sc.setMode(5)
sc.setOrder([2, 0, 0, 0, 14])  # [h, i, j, k, l] — same as 2·Qx = 14 in mode 3
result = sc.getResonanceDrivingTerms()
```

---

### Detuning coefficients

```python
sc.setParameters(intensity=1.6e11, bunchLength=0.09, ro=1.5347e-18,
                 emittance_x=2.5e-6, emittance_y=2.5e-6, dpp_rms=1.1e-4)
sc.prepareData("twiss.tfs")
sc.setMode(3)
sc.setOrder([2, 0, "any"])   # 2nd-order horizontal — amplitude detuning

dQ = sc.getDetuning()
print(f"Detuning coefficient : {dQ:.4e}")
```

---

### Loading from X-suite

If your lattice is tracked with X-suite, pass the Twiss dictionary directly:

```python
import xtrack as xt

line = xt.Line.from_json("lhc.json")
line.build_tracker()
twiss = line.twiss()         # returns a TwissTable dict-like object

sc = PySCRDT()
sc.setParameters(intensity=1.6e11, bunchLength=0.09, ro=1.5347e-18,
                 emittance_x=2.5e-6, emittance_y=2.5e-6, dpp_rms=1.1e-4)
sc.loadTwissFromXsuite(twiss)   # replaces prepareData()

sc.setMode(3)
sc.setOrder([4, 0, 28])
print(sc.getResonanceDrivingTerms())
```

The dictionary must contain the keys `s`, `betx`, `bety`, `dx`, `dy`, `mux`,
`muy`, `circumference`, `qx`, `qy`, and `particle_on_co`.

---

### Reading parameters from a file

```python
sc.readParameters("beam_params.txt")
```

The file format is one entry per line:

```
intensity   = 1.6e11
bunchLength = 0.09
emittance_x = 2.5e-6
emittance_y = 2.5e-6
dpp_rms     = 1.1e-4
```

---

## API reference

| Method | Description |
|---|---|
| `setParameters(**kwargs)` | Set beam parameters (see table below). |
| `readParameters(file)` | Read beam parameters from a plain-text file. |
| `updateParameters(**kwargs)` | Update individual entries in the parameter dict. |
| `getParameters()` | Return the current parameter dict. |
| `prepareData(twissFile)` | Load and interpolate a MAD-X Twiss file. |
| `loadTwissFromXsuite(twissDict)` | Load Twiss data from an X-suite dict. |
| `setMode(mode)` | Set resonance description mode: `3` or `5`. |
| `setOrder(args)` | Set resonance order: `[m, n, l]` (mode 3) or `[h, i, j, k, l]` (mode 5). |
| `getMode()` | Return the active mode. |
| `getOrder()` | Return the active resonance order as a tuple. |
| `getResonanceDrivingTerms(feedDown=False)` | Return `{'RDT', 'Amplitude', 'Phase'}`. |
| `getDetuning()` | Return the accumulated detuning coefficient (complex). |
| `getKsc()` | Return the space charge perveance K_sc. |
| `getPotential()` | Return the symbolic SC potential (SymPy expression). |
| `getWorkingPoint()` | Return the tunes `(Qx, Qy)` read from the Twiss file. |
| `checkWriting()` | Return valid parameter key names for `setParameters` / `updateParameters`. |

### Beam parameters

| Key | Description | Default |
|---|---|---|
| `intensity` | Bunch population [particles] | `41e10` |
| `bunchLength` | RMS bunch length [m] | `5.96` |
| `ro` | Classical particle radius [m] | `1.5347e-18` (proton) |
| `emittance_x` | Normalised horizontal emittance [m·rad] | `2e-6` |
| `emittance_y` | Normalised vertical emittance [m·rad] | `1.1e-6` |
| `dpp_rms` | RMS momentum spread Δp/p | `0.5e-3` |
| `dpp` | Single-particle Δp/p (feed-down) | `0.0` |
| `bF` | Bunching factor (replaces `bunchLength` if set) | `None` |
| `harmonic` | Harmonic number / number of buckets | `1` |

The parameters `b` (relativistic β) and `g` (Lorentz γ) are derived automatically
from the Twiss file and can be overridden with `updateParameters`.

---

## Contact

Foteini Asvesta — `foteini.asvesta@cern.ch`
