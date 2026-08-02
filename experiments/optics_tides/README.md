# Optics + tides (central Earth-scale observer)

## Optics procedure

1. Segment time by which fairy is farthest from the central observer.
2. In each segment, min visual distance (angular) → brightness-swap point.
3. **1 Sun + 3 Moons**: Sun self-luminous (`L_sun` fixed); moons reflect via **`starry.Map(...).flux`** with shared albedo `--A`.
4. **Evenness = period variance of `F_total`**, not per-body albedo:
   - `uneven`: Sun role rotates at swap points → higher `F_var` / `F_cv`
   - `uniform`: Sun identity fixed → lower period variance

Needs `pip install -r requirements-optics.txt` (sets `THEANO_FLAGS=...cxx=` for NumPy 2).
Default `--illumination near` (self-luminous Sun at fairy); use `far` for parallel-ray phase curves.

```powershell
# uneven (role rotate)
.\.venv\Scripts\python.exe experiments\optics_tides\run_central_optics.py `
  --seed experiments\output\continuation_n4_cycle\trial_26149\state_Mc_6.875958e-02.json `
  --m-c 0.06875958 --periods 2 --A 0.25 --evenness uneven

# uniform (fixed Sun)
.\.venv\Scripts\python.exe experiments\optics_tides\run_central_optics.py `
  --seed experiments\output\continuation_n4_cycle\trial_26149\state_Mc_6.875958e-02.json `
  --m-c 0.06875958 --periods 2 --A 0.25 --evenness uniform
```

Tides: reserved.
