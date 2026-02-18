# CHANGELOG

## v1.1.0 — Modernisation of PySCRDT

**Audience:** This section is written specifically for the original author to review any
change that touches calculation logic. Purely structural changes (project layout, packaging,
type annotations, logging, test suite) are summarised briefly at the end.

---

### Calculation / physics changes (please review)

The items below are the only places where the numerical computation was altered.
Each entry states what the original code did, what the new code does, and the
rationale. Where the result should be identical the reasoning is given; where there
is any residual doubt it is flagged explicitly.

---

#### 1. `potential()` — redundant second `.doit()` call removed

**File/lines:** `_core.py : potential()`

**Original:**
```python
res = sy.integrate(sterm, (self.t, 0, sy.oo)).doit()
result = res.doit()          # ← called a second time
self.V = sy.simplify(result)
```

**New:**
```python
res = sy.integrate(sterm, (self.t, 0, sy.oo)).doit()
self.V = sy.simplify(res)
```

**Why:** `sy.integrate(...).doit()` already forces full evaluation of the integral.
Calling `.doit()` on the already-evaluated expression is documented as a no-op in
SymPy — it returns the same object. The intermediate variable `result` was
therefore identical to `res`.

**Verdict:** Numerically identical. However, if you have ever observed a case where
the integral evaluates lazily and the second `.doit()` was necessary to get a
simplified form, please re-add it.

---

#### 2. `prepareData()` — dispersion columns scaled before interpolation (not inside it)

**File/lines:** `_core.py : prepareData()`

**Original (inline, inside `np.interp`):**
```python
data2[:,3] = np.interp(s, data[:,0], self.parameters['b'] * data[:,3])
data2[:,4] = np.interp(s, data[:,0], self.parameters['b'] * data[:,4])
```

**New (in-place on raw array before calling `_interpolate`):**
```python
beta = self.parameters["b"]
data[:, 3] *= beta   # DX
data[:, 4] *= beta   # DY
self._interpolate(data)
# inside _interpolate:
data2[:, 3] = np.interp(s, data[:, 0], data[:, 3])
data2[:, 4] = np.interp(s, data[:, 0], data[:, 4])
```

**Why:** The multiplication was moved out of the `np.interp` call to allow both
`prepareData` and `loadTwissFromXsuite` to share the same interpolation routine
(`_interpolate`).

**Verdict:** Numerically identical. `np.interp` is a linear operation:
`np.interp(s, x, β·y) ≡ β · np.interp(s, x, y)`.
The beta-scaled dispersion values stored in `self.data[:,3]` and `self.data[:,4]`
are the same in both versions.

**Note on mutation:** In the new code `data[:, 3] *= beta` mutates the local
`data` array that was returned by `np.loadtxt`. Because `data` is a local variable
inside `prepareData`, this has no side effects.

---

#### 3. `loadTwissFromXsuite()` — same dispersion beta scaling change

**File/lines:** `_core.py : loadTwissFromXsuite()`

**Original (inline, inside `np.interp`):**
```python
data2[:,3] = np.interp(s, twissTableXsuite['s'], self.parameters['b'] * twissTableXsuite['dx'])
data2[:,4] = np.interp(s, twissTableXsuite['s'], self.parameters['b'] * twissTableXsuite['dy'])
```

**New (applied when constructing the staging array):**
```python
beta = self.parameters["b"]
raw = np.column_stack([
    s_raw,
    twissTableXsuite["betx"],
    twissTableXsuite["bety"],
    beta * np.asarray(twissTableXsuite["dx"]),   # DX
    beta * np.asarray(twissTableXsuite["dy"]),   # DY
    twissTableXsuite["mux"],
    twissTableXsuite["muy"],
    np.zeros(len(s_raw)),
])
self._interpolate(raw)
```

**Verdict:** Numerically identical for the same reason as item 2 above.

---

#### 4. Interpolated data array — column 7 initialisation (`+=` → `=`)

**File/lines:** `_core.py : _interpolate()` (was inline in both `prepareData` and `loadTwissFromXsuite`)

**Original:**
```python
data2 = np.zeros((100000, 8))
# ... other columns filled ...
data2[:,7] += self.parameters['C'] / len(s)
```

**New:**
```python
data2 = np.zeros((100_000, 8))
# ...
data2[:, 7] = C / len(s)
```

**Why:** `data2` is freshly allocated with `np.zeros`, so every element is 0.
`0 + x` is `x`, therefore `+= x` is equivalent to `= x`.

**Verdict:** Numerically identical.

---

#### 5. Interpolated data array — column 0 (`s`) assigned first instead of last

**File/lines:** `_core.py : _interpolate()`

**Original:** `data2[:,0] = s` was the **last** assignment (after all other columns).

**New:** `data2[:, 0] = s` is the **first** assignment.

**Why:** Purely cosmetic reordering during the shared-interpolation refactor.

**Verdict:** No effect. Each column assignment is independent; reordering them
cannot affect the values stored. `s` is fully defined before any column is written.

---

#### 6. `resonanceDrivingTerms()` — final accumulation: `sum()` → `ndarray.sum()`

**File/lines:** `_core.py : resonanceDrivingTerms()`

**Original:**
```python
self.rdt = sum(self.rdt_s)     # Python built-in sum
```

**New:**
```python
self.rdt = self.rdt_s.sum()    # NumPy ndarray method
```

**Why:** `self.rdt_s` is a NumPy complex array. Calling Python's built-in `sum()`
on a NumPy array iterates element-by-element in Python, accumulating from the
integer `0`. Calling `.sum()` uses NumPy's internal pairwise reduction which is
faster and the standard NumPy idiom.

**Verdict:** Results agree to within floating-point rounding. Because the two
reductions may accumulate in a different order, the last 1–2 ULPs of the
complex result can differ. For a 100 000-element sum this is negligible for
any physically meaningful use (amplitudes, phases), but it is technically a
change. If reproducibility of bit-exact results against earlier runs is
important, restore `sum(self.rdt_s)`.

---

#### 7. `detuning()` — same accumulation change

**File/lines:** `_core.py : detuning()`

**Original:**
```python
self.rdt_d = sum(self.rdt_s_d)
```

**New:**
```python
self.rdt_d = self.rdt_s_d.sum()
```

**Verdict:** Same as item 6.

---

#### 8. `calculateFactor()` — intermediate variable `det3` removed

**File/lines:** `_core.py : calculateFactor()`

**Original (three cases, e.g. `m==0`):**
```python
det1 = sy.cos(self.fy) ** abs(self.n)
det3 = det1.rewrite(sy.exp)      # intermediate variable
det2 = sy.expand(det3)
```

**New:**
```python
det1 = sy.cos(self.fy) ** abs(self.n)
det2 = sy.expand(det1.rewrite(sy.exp))   # collapsed into one line
```

**Why:** `det3` held the result of `det1.rewrite(sy.exp)` and was used in exactly
one place immediately after. It was an unnecessary intermediate variable.

**Verdict:** Numerically identical. The same SymPy expression is passed to
`sy.expand()` in both cases.

---

#### 9. `resonanceDrivingTerms()` mode-5 — local aliases for repeated sub-expressions

**File/lines:** `_core.py : resonanceDrivingTerms()`

**Original:** the expressions `(self.h+self.i)`, `(self.j+self.k)`,
`(self.h-self.i)`, `(self.j-self.k)` were written out inline in every term of
the amplitude and phase formulas.

**New:** these are computed once into named locals before the formula:
```python
m_eff = self.h + self.i    # total H order (amplitude exponent)
n_eff = self.j + self.k    # total V order (amplitude exponent)
m_ph  = self.h - self.i    # H phase advance coefficient
n_ph  = self.j - self.k    # V phase advance coefficient
```

**Verdict:** Purely cosmetic. The mathematical content of each formula is unchanged.

---

#### 10. `reIndexing()` — `self.dictionary` changed to a local variable

**File/lines:** `_core.py : reIndexing()`

**Original:**
```python
self.dictionary = {}        # stored as instance attribute
for i in factor.keys():
    ...
return self.dictionary
```

**New:**
```python
dictionary: dict = {}       # local variable
for key, val in factor.items():
    ...
return dictionary
```

**Why:** `self.dictionary` was never part of the documented interface and was
never read back from the instance by any other method. Storing it as an instance
attribute caused the dictionary from the most recent `calculateFactor` call to
persist on the object indefinitely. Making it local is safer and avoids confusion.

**Verdict:** No effect on calculation results. If you relied on inspecting
`instance.dictionary` after a `calculateFactor` call for debugging, that is no
longer available.

---

#### 11. `setOrder()` — mode/length guard logic rewritten

**File/lines:** `_core.py : setOrder()`

**Original logic (for `len(args)==3`):**
```python
if self.mode == 3 or self.mode == None:   # allow if mode is 3 OR unset
    # assign m, n, l
    self.mode = 3
else:
    raise IOError(...)   # mode is 5 → error
```

**New logic:**
```python
if self.mode == 5:
    raise ValueError(...)   # mode is 5 → error
# otherwise assign m, n, l
self.mode = 3
```

**Verdict:** Logically equivalent. Both versions:
- Accept a 3-element list when `mode` is None or 3, setting `mode = 3`.
- Raise when `mode` is 5 and a 3-element list is given.
- Accept a 5-element list when `mode` is None or 5, setting `mode = 5`.
- Raise when `mode` is 3 and a 5-element list is given.

The new form is cleaner because it states the forbidden case explicitly rather
than the allowed case.

---

### Non-calculation changes (summary only)

These changes do not affect physics results and are included for completeness.

| Area | Change |
|---|---|
| Project layout | `PySCRDT/PySCRDT.py` → `src/PySCRDT/_core.py` (src layout) |
| Packaging | `setup.py` removed; `pyproject.toml` added (PEP 517/518) |
| Python version | Minimum raised to 3.9; Python 2 artifacts removed (`from __future__ import …`, `class PySCRDT(object):`) |
| Critical Python 3 bug | `kwargs.iteritems()` → `kwargs.items()` in `updateParameters()` — the old code crashed on every Python 3 call |
| Type checking | `IOError` for validation replaced by `ValueError`/`TypeError`; `type(x) is int` → `isinstance(x, int)` |
| Type annotations | All public methods annotated; resonance order attributes (`m`, `n`, `h`, `i`, `j`, `k`, `l`) declared in `__init__` |
| Logging | `print()` calls replaced with `logging.getLogger(__name__)` at INFO level |
| Error messages | All exception strings rewritten to remove the `# PySCRDT::method:` prefix and use f-strings |
| Refactor | Beta-function interpolation factored into private `_interpolate()` (eliminates code duplication between `prepareData` and `loadTwissFromXsuite`) |
| `readParameters()` | `for i in enumerate(params): params[i[0]][...]` → `for row in params: row[...]` (cleaner unpacking, same logic) |
| `prepareData()` | `for line in enumerate(f.readlines()): line[0], line[1]` → `for line_nr, line in enumerate(f):` (same logic) |
| Tests | New `tests/test_core.py` with 21 smoke tests covering instantiation, parameter management, mode/order setting, and `updateParameters` |
| Static analysis | `pyrightconfig.json` added to suppress SymPy metaclass false positives in Pylance |
