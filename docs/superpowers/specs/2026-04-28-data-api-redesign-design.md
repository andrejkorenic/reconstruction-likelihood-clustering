# RLC tabular time-series data API — design

## Context

RLC's current time-series loaders carry project-specific assumptions that bleed "calcium" semantics into a repo that should be neutral and reusable:

- **`parquet_timeseries_loader`** — docstring literally says "calcium ingestion output". Hardcoded regex `^t_\d+$` for sample columns. Labels always zero (`y_all = np.zeros(N)`). Fixed 80/10/10 split. Only global MinMax normalisation. `feat_dim = 1` only.
- **`csv_timeseries_loader`** — `--csv_meta_cols` arbitrary "drop-first-N" convention. Same fixed split, same single normalisation, same univariate-only constraint.
- **Image loaders** — closed list of benchmark datasets (MNIST, CIFAR, SVHN, etc.). Anyone with their own image data must subclass `base_load_data` and edit the dispatch in `data_loader_instances.load_dataset()`.

The user's intent (recorded in `TODO.md` under "Next workstream: generalize the data-loading API"): RLC must not embed any project-specific data convention. A clean, well-documented API for time-series data — and later for images — is required so that any downstream consumer (calcium_analysis is the first, but not the only) brings its own data without writing new loader classes or fighting hardcoded conventions.

This spec covers **time-series only**. The image API is a separate future spec — same brainstorm decided on phased delivery (TS first because it has a concrete consumer in calcium; image generalisation is currently hypothetical).

The benchmark TS loaders (`ecg5000_loader`, `synthetic_timeseries_loader`) are out of scope and remain untouched. They serve as the TS-side analogue of MNIST: known reference datasets, useful for sanity-checking the codebase.

## Decisions

Locked during brainstorm (2026-04-28):

- **One unified loader, named `tabular_timeseries`**, replaces both `csv_timeseries` and `parquet_timeseries`. Reads csv/tsv/parquet/npy via the same code path with a single set of `--ts_*` CLI flags.
- **Format auto-detected** from file extension; explicit `--ts_format` overrides only when extension and content disagree.
- **Column selection** via single `--ts_value_cols` flag. Vrednost koja sadrži zarez tretira se kao explicit lista; vrednost bez zareza tretira se kao regex pattern. `--ts_value_cols` je obavezan za csv/parquet, ignoriše se za npy.
- **Optional label column** via `--ts_label_col`; default None gives all-zero labels (current behaviour). String labels pretvaraju se u integer ID-ove kroz `pandas.factorize`.
- **Configurable split**: `--ts_split` (default `"0.8/0.1/0.1"`) za single-fajl auto-split, ili `--ts_train_path` / `--ts_val_path` / `--ts_test_path` za eksplicitno pre-split.
- **Four normalisation modes** kroz `--ts_normalise`: `global_minmax` (default, trenutno calcium ponašanje), `per_sample_minmax`, `zscore`, `none`. Statistika se uvek računa **iz train splita** i primenjuje na sva tri (sprečava test-set leakage).
- **Univariate only** (`feat_dim = 1`). Multivariate se ne podržava u ovom radu — ostaje u `TODO.md` za budući spec.
- **Stari `csv_timeseries` i `parquet_timeseries` se uklanjaju** zajedno sa svojim klasama i testovima. Calcium-side migracija (update njegovih `run.py` poziva) je out-of-scope ovde — radi se zasebno posle merge-a `consolidate-clean-api → main`.
- **Branch:** rad ide na `data-api-redesign` (off `consolidate-clean-api`). Ne push, ne merge dok ne završi review.
- **Co-Authored-By:** `Co-Authored-By: Claude Opus 4.7 (1M context)` na svakom commit-u (bez email-a, bez angle brackets).

## CLI surface

Svi novi flagovi imaju prefiks `--ts_*` da budući image API može koristiti `--img_*` bez kolizije.

| Flag | Tip | Default | Šta radi |
|---|---|---|---|
| `--ts_path` | str (path) | None | Putanja do single-fajl dataset-a (csv/tsv/parquet/npy). Mutually exclusive sa `--ts_*_path` trio. |
| `--ts_train_path` | str (path) | None | Putanja do train fajla. Mora se zadati zajedno sa val + test putanjama. |
| `--ts_val_path` | str (path) | None | Putanja do val fajla. |
| `--ts_test_path` | str (path) | None | Putanja do test fajla. |
| `--ts_format` | enum `{csv, parquet, npy, auto}` | `auto` | Kad je `auto`, format se određuje iz ekstenzije: `.csv`/`.tsv` → csv, `.parquet` → parquet, `.npy` → npy. Override za nestandardne ekstenzije. |
| `--ts_csv_sep` | str | `,` | Separator za csv format. Za TSV proslediti `\t`. Ignoriše se za parquet/npy. |
| `--ts_value_cols` | str | None | Selektor sample kolona. Bez zareza → regex; sa zarezom → explicit lista (npr. `"t_0,t_1,t_2"`). Obavezan za csv/parquet; ignoriše se za npy (ceo niz je `(N, T)`). |
| `--ts_label_col` | str | None | Ime kolone sa labelama. None → sve nule. Strings se konvertuju u integer ID kroz `pd.factorize`. Ignoriše se za npy. |
| `--ts_split` | str `"a/b/c"` | `"0.8/0.1/0.1"` | Train/val/test ratio za single-fajl mode. Ignoriše se ako je trio puta dat. Mora sumirati u 1.0 (±1e-6). |
| `--ts_normalise` | enum | `global_minmax` | `global_minmax`, `per_sample_minmax`, `zscore`, ili `none`. Statistika iz train-a, primenjena na sva tri splita. |

### Validacioni invariants (fail-loud)

- `--ts_path` i bilo koja od `--ts_train_path`/`--ts_val_path`/`--ts_test_path` istovremeno → exit sa porukom "specify either --ts_path OR all three of --ts_*_path, not both".
- Trio puta: ako su data 1 ili 2 puta od 3 → exit sa porukom koja kaže koji fali.
- csv/parquet bez `--ts_value_cols` → exit sa porukom koja navodi prvih 5 kolona u fajlu.
- `--ts_value_cols` regex koji ne matchuje nijednu kolonu → exit sa listom svih kolona u fajlu da korisnik može da popravi pattern.
- `--ts_value_cols` lista u kojoj jedna ili više kolona ne postoje u fajlu → exit sa listom kolona koje fale.
- `--ts_label_col` koji ne postoji u fajlu → exit sa porukom.
- npy fajl koji nije 2D → exit ("expected (N, T), got shape ...").
- `--ts_split` ratio koji ne sumira u 1.0 → exit.

Sve ove validacije važe i za single-fajl mode i za pre-split mode (gde se primenjuju nezavisno na svaki od 3 fajla).

## Loader implementation contract

`tabular_timeseries_loader.load_dataset()` se izvršava sledećim koracima:

1. **Validacija ulaza** (vidi gore).
2. **Format detekcija**: ako je `--ts_format auto`, mapira ekstenziju → format.
3. **Učitavanje sirovih podataka**:
   - csv/tsv: `pd.read_csv(path, sep=args.ts_csv_sep)` → DataFrame.
   - parquet: `pd.read_parquet(path)` → DataFrame.
   - npy: `np.load(path)` → mora biti 2D `(N, T)`.
4. **Selekcija kolona** (csv/parquet samo):
   - Ako `--ts_value_cols` sadrži zarez: `cols = pattern.split(',')`. Validira da svaka kolona postoji u DataFrame-u.
   - Inače: `t_pat = re.compile(pattern); cols = [c for c in df.columns if t_pat.match(c)]`. Validira da je `len(cols) > 0`.
   - Sortira `cols` po imenu (deterministički redosled — bitno za reproducibilnost preko različitih pandas verzija).
   - `x = df[cols].to_numpy(dtype=np.float32)` → `(N, T)`.
5. **Labela**:
   - Ako `--ts_label_col` dat: `y = pd.factorize(df[label_col])[0].astype(np.int64)`.
   - Inače: `y = np.zeros(len(x), dtype=np.int64)`.
6. **Split**:
   - Pre-split mode (3 fajla): koraci 3–5 se primenjuju nezavisno na svaki fajl. `(x_train, y_train)`, `(x_val, y_val)`, `(x_test, y_test)` direktno iz fajlova. **Validacija**: `T` (broj sample kolona) mora biti isti u sva tri splita.
   - Single-fajl mode: deterministic shuffle preko `np.random.default_rng(args.seed)`, isečci po `--ts_split` ratio-u.
7. **Normalizacija** (`--ts_normalise`):
   - `global_minmax`: `x_min, x_max = x_train.min(), x_train.max()`; svaki split `(x - x_min) / (x_max - x_min + 1e-7)`.
   - `per_sample_minmax`: za svaki red u svakom splitu, `(x - x.min()) / (x.max() - x.min() + 1e-7)`.
   - `zscore`: `mu, sigma = x_train.mean(), x_train.std()`; svaki split `(x - mu) / (sigma + 1e-7)`.
   - `none`: prosleđuje x kao što je. Korisnik je odgovoran za smislenu normalizaciju (Beta likelihood pretpostavlja `[0, 1]`).
8. **Args runtime fields** (in-place mutacija na `args`, isti pattern kao postojeći loader-i):
   - `args.seq_len = T`
   - `args.feat_dim = 1`
   - `args.input_size = [1, T]`
   - `args.input_type = 'continuous'`
   - `args.use_logit = False`
   - `args.dynamic_binarization = False`
   - `args.training_set_size = len(x_train)`
9. **Reshape u flat**: `x_train, x_val, x_test` se reshape-uju iz `(N, T)` u `(N, 1*T)` (trivijalno, ali zadržava paritet sa drugim flat 1-D loader-ima).
10. **`post_processing()`** iz `base_load_data` (gradi `TensorDataset` sa indices za ExemplarVAE LOO masking, `DataLoader`-e, pokreće VampPrior init).

Povratna vrednost: `(train_loader, val_loader, test_loader, args)` — isti tip kao trenutni `csv_timeseries_loader.load_dataset()`.

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `utils/load_data/timeseries_loader.py` | modify | Dodati `tabular_timeseries_loader` klasu (~150 linija). Ukloniti `csv_timeseries_loader` i `parquet_timeseries_loader` klase. `ecg5000_loader` i `synthetic_timeseries_loader` ostaju netaknute. |
| `utils/load_data/data_loader_instances.py` | modify | Ukloniti `elif args.dataset_name == 'csv_timeseries':` i `elif args.dataset_name == 'parquet_timeseries':`. Dodati `elif args.dataset_name == 'tabular_timeseries':`. |
| `run.py` | modify | Dodati ~10 novih `--ts_*` argparse flagova u nove ili postojeće argument grupe. Ukloniti stare `--csv_path`, `--csv_meta_cols`, `--parquet_path` (zamenjeni `--ts_path` + `--ts_value_cols`). |
| `analyze.py` | modify | `--export_latents` koristi `--parquet_override` ili `config.parquet_path`/`config.csv_path`. Migrirati na: `config.ts_path` + `config.ts_value_cols` + `--ts_path_override` (renamed iz `--parquet_override`). API.md update reflektuje. |
| `unit_tests/test_tabular_timeseries.py` | create | Pure-function testovi (rade na sintetičkim DataFrame-ovima u `tmp_path`). Pokriva sve flagove + validacione greške (vidi Testing). |
| `unit_tests/test_csv_timeseries.py` | delete | Pokriveno novim testom. |
| `unit_tests/test_parquet_timeseries.py` | delete | Pokriveno novim testom. |
| `unit_tests/test_export_latents.py` | modify | Update flagove u subprocess pozivima sa `--csv_path` na `--ts_path`/`--ts_value_cols`. |
| `unit_tests/test_export_pseudo_prototypes.py` | modify | Isto kao gore. |
| `unit_tests/test_ood_recon_nll.py` | modify | Isto. |
| `unit_tests/test_ood_pseudo_recon.py` | modify | Isto. |
| `unit_tests/test_cli_contract.py` | modify | Update fixture training poziv na nove flagove. |
| `API.md` | modify | Nova sekcija "Tabular time-series — bring your own data" sa kompletnim primerima (csv, parquet, npy, pre-split, sa labelama, sa custom normalizacijom). Ukloniti reference na `csv_timeseries`/`parquet_timeseries`. |
| `TODO.md` (gitignored) | modify | Popuniti "Conclusions" sekciju ispod "Next workstream: generalize..." sa donesenim odlukama. Dodati napomenu da calcium-side migracija je out-of-scope ovde. |

## Testing

`unit_tests/test_tabular_timeseries.py` pokriva (sve testovi rade na sintetičkim podacima u `tmp_path`, brzi i deterministički):

### Pozitivni put

- csv ulaz sa regex `--ts_value_cols`.
- csv ulaz sa explicit listom `--ts_value_cols "a,b,c"`.
- tsv ulaz preko `--ts_csv_sep "\t"`.
- parquet ulaz sa regex.
- npy ulaz (2D `(N, T)` direktno).
- Auto-detekcija formata iz ekstenzije za sva 4 formata.
- Override `--ts_format` kad ekstenzija je nestandardna.
- `--ts_label_col` sa string vrednostima (factorize).
- `--ts_label_col` sa integer vrednostima.
- 3-puta pre-split mode (csv).
- 3-puta pre-split mode sa različitim formatima u kombinaciji (csv + parquet) — graceful (svaki fajl se učita po sopstvenom formatu).
- Single-fajl split sa custom ratio-om (`--ts_split "0.7/0.15/0.15"`).
- Determinizam: dva run-a sa istim `--seed` daju isti train/val/test split.

### Normalizacije

Za svaki od 4 mode-a, sintetički input gde se **može tačno proveriti rezultat**:
- `global_minmax`: train ima poznate min/max; verifikuj da se test ne koristi za izračun (ako je test min < train min, posle normalizacije će test biti negativan — to je očekivano i to ovaj test demonstrira).
- `per_sample_minmax`: svaki red posle nezavisno u `[0, 1]`.
- `zscore`: train ima `mean ≈ 0, std ≈ 1` posle (do epsilon), test može odstupati.
- `none`: vrednosti su identične input-u.

### Validacione greške (sve treba da pucaju sa jasnom porukom)

- `--ts_path` + `--ts_train_path` istovremeno.
- Pre-split sa 1 ili 2 puta dat (umesto 3).
- csv bez `--ts_value_cols`.
- Regex koji ne matchuje nijednu kolonu — poruka navodi sve kolone u fajlu.
- Lista kolona u kojoj jedna ne postoji.
- `--ts_label_col` koji ne postoji.
- npy 1D ili 3D shape.
- `--ts_split` ratio koji ne sumira u 1.0.
- Pre-split fajlovi sa različitim T (broj sample kolona) → fail.

Plan: ~25 test funkcija, ukupno ~400 linija. Stari `test_csv_timeseries.py` (132 linije, samo srećan slučaj) i `test_parquet_timeseries.py` se brišu.

## Migration notes

### Unutar RLC repo-a

Postojeći testovi koji subprocess-uju `run.py` da bi trenirali tiny model imaju hardkodovane stare flagove:

- `--dataset_name csv_timeseries --csv_path ... --csv_meta_cols 0` → `--dataset_name tabular_timeseries --ts_path ... --ts_value_cols "<regex_or_list>"`

Affected files (već enumerirani gore u file structure):
- `unit_tests/test_export_latents.py`
- `unit_tests/test_export_pseudo_prototypes.py`
- `unit_tests/test_ood_recon_nll.py`
- `unit_tests/test_ood_pseudo_recon.py`
- `unit_tests/test_cli_contract.py`

Update se radi u istom plan-u, kao deo zadatka koji uvodi novu loader klasu.

### Calcium-side (out of scope)

Calcium-ovi `train_*.py` skripti (verovatno `train_seg1.py`, `train_seg1_als.py`, ako postoje) zovu RLC kroz `subprocess` sa starim flagovima. Posle merge-a `consolidate-clean-api → main` i posle merge-a ovog rada (`data-api-redesign → main`), calcium će tražiti separate task da update-uje svoje subprocess pozive na novu CLI površinu. Spec to **eksplicitno ne pokriva** — nije RLC-ov posao.

Update tipa:
```python
# Before
subprocess.run(["uv", "run", "python", "run.py",
                "--dataset_name", "parquet_timeseries",
                "--parquet_path", str(parquet),
                ...])

# After
subprocess.run(["uv", "run", "python", "run.py",
                "--dataset_name", "tabular_timeseries",
                "--ts_path", str(parquet),
                "--ts_value_cols", r"^t_\d+$",
                ...])
```

Mehanički, zahteva grep + zamenu u 1–3 mesta. Beleška se zapisuje u calcium-ov `TODO.md` (kao part-of-this work).

### `analyze.py --parquet_override`

Trenutni flag `--parquet_override` cilja parquet specifično. U novoj API-ji rename u `--ts_path_override` (pošto sad može biti csv/parquet/npy). Update API.md i sve testove koji ga koriste.

## Out of scope

- **Image API** — sledeći spec u zasebnom ciklusu.
- **Multivariate TS** (`feat_dim > 1`) — zahteva i izmene u TimeSeriesVAE arhitekturi (decoder ulaz/izlaz oblika); ne YAGNI dok ne bude realnog konzumenta. Ostaje u `TODO.md`.
- **Plugin/registration sistem** — calcium je CLI subprocess konzument, ne import konzument; trenutni dispatch radi i ne prepravljamo bez razloga.
- **Refaktor `ecg5000_loader` / `synthetic_timeseries_loader`** — funkcionalni, paralela sa MNIST-om na image strani; ne diramo.
- **Refaktor `base_load_data` `args` mutacije** — postojeći side-effect pattern radi, novi loader ga prati. Bilo kakav redizajn `LoaderResult` dataclass-a je posebno.
- **Calcium-side migracija** — RLC završi svoj posao, calcium update se radi posebno.
- **Backward-compat alias-i** za `csv_timeseries`/`parquet_timeseries` — calcium je jedini konzument; lakše je da calcium update-uje svoje pozive nego da RLC nosi deprecirana imena.

## Verification

Posle implementacije:

1. `uv run pytest unit_tests/ -q -m "not slow"` — sve zelene (postojećih 233 + ~25 novih iz `test_tabular_timeseries`, minus uklonjenih iz `test_csv_timeseries` + `test_parquet_timeseries`).
2. `grep -rn "csv_timeseries\|parquet_timeseries\|csv_path\|parquet_path\|csv_meta_cols\|parquet_override" --include="*.py"` — prazno (osim možda u istorijskim docstring-ovima ili u out-of-scope `ecg5000`/`synthetic`).
3. Manual smoke: trenirati tiny model preko nove CLI površine na calcium parquet fajlu (`segment1_active.parquet`), `--ts_value_cols "^t_\d+$"`, verifikovati da `MODEL_DIR` izlazi i da `analyze.py --export_pseudo_prototypes` i dalje radi.
4. `cat API.md | grep -E "tabular_timeseries|--ts_"` — postoji nova sekcija sa kompletnim primerima.
5. `git log --oneline consolidate-clean-api..HEAD` — čista sekvenca commit-a sa Co-Authored-By trailer-om.
6. Branch `data-api-redesign` pushovan na origin **bez merge-a** dok user ne odobri.
