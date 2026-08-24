# 🧭 FORK_GUIDE.md — Pilotage du fork « Le planificateur d'inférence locale »

> **⚠️ CE FICHIER EST LA MÉMOIRE DU PROJET.** Il survit aux pertes de contexte.
> Toute session (humaine ou agent) DOIT commencer par le lire en entier, puis suivre le protocole §0.
> Il est mis à jour APRÈS CHAQUE item terminé (cocher la case + entrée journal §6 + commit).
> Contexte complet du projet : `../RAPPORT_llmfit_audit_fork.md` (audit initial, à lire au moins une fois).

---

## 0. PROTOCOLE DE DÉMARRAGE DE SESSION (exécuter dans l'ordre, sans exception)

1. Lire CE fichier en entier.
2. `cd /home/jose/internship/fast/llmfit && git status && git log --oneline -5` — vérifier qu'on est sur `main`, propre.
3. Si `git status` n'est pas propre : terminer/annuler proprement l'item en cours AVANT d'en commencer un autre.
4. Trouver le **premier item non coché `[ ]`** de la roadmap §4.
5. Relire les specs techniques de cet item dans §4 + les pièges §7.
6. Lire `AGENTS.md` à la racine du repo (conventions upstream) avant d'écrire du code.
7. Implémenter l'item **jusqu'à Done** : code + tests unitaires verts + `cargo clippy` propre + `cargo fmt`.
8. Commit (style conventionnel anglais, cf. §2), puis :
   - cocher la case `[x]` dans ce fichier,
   - ajouter une entrée datée au journal §6,
   - commit `docs(guidance): <item>` incluant le guide,
   - `git push origin main`.

**Règle d'or : jamais plus d'un item en cours. Un item commencé doit être fini ou explicitement marqué BLOQUÉ dans §6 avec la raison.**

---

## 1. Objectif & positionnement

Transformer llmfit (filtre de faisabilité binaire « ça passe / ça casse ») en **planificateur d'inférence locale** :

```
Qwen3-235B-A22B sur ta machine (RTX 3090 24Go + DDR5 96Go) :
→ Q4_K_M, contexte 16k, 12 experts offloadés en RAM, 20/48 couches GPU
→ débit attendu : 11–14 tok/s (génération), TTFT ~0.8s
→ marge VRAM : 1,2 Go
→ commande : llama-server -m ... --n-cpu-moe 12 -ngl 20 -c 16384 -fa
```

Différenciateur défendable : personne ne sort aujourd'hui une **commande complète prête à lancer** avec débit prédicté ± incertitude. Espace vide entre `llama.cpp --fit` (exécution seule) et les calculateurs web (estimation seule).

## 2. Règles d'autonomie

- **Qualité d'abord** : chaque fix testé avant de passer au suivant. Pas de test skippé (`#[ignore]` interdit sauf raison documentée dans §6).
- **Commits** : style conventionnel anglais comme upstream (`fix(fit): ...`, `feat(gguf): ...`). 1+ commit par item. Push vers `origin main` après chaque milestone (fin de version).
- **Upstream** (`remote upstream` = AlexsJones/llmfit) : **IGNORER**. Pas de fetch/merge/rebase. Divergence assumée (rapport §Risques).
- **Ne jamais casser la baseline** : `cargo build --release && cargo test` vert avant/après chaque commit.
- **Pas de commentaires décoratifs** ; commentaires uniquement quand la physique/l'intention n'est pas évidente.
- Langue du code/docs utilisateur : anglais (cohérence upstream). Ce guide : français.
- En cas de doute d'implémentation : choisir l'honnêteté (fourchette large + hypothèse documentée) plutôt que la précision illusoire.

## 3. État de l'environnement (validé 2026-08-23)

| Élément | Valeur |
|---|---|
| Repo local | `/home/jose/internship/fast/llmfit` |
| origin | `git@github.com:THEGENTLEMAN31/llmfit.git` (SSH OK) |
| upstream | `https://github.com/AlexsJones/llmfit.git` — ignoré |
| Branche de travail | `main` |
| HEAD de départ | `3f44fd3` (= v1.1.10, même arbre que l'audit) |
| Toolchain | cargo/rustc 1.97.1, Arch Linux |
| Baseline build/test/clippy | ✅ build 3m23s · **565 tests verts** (1 ignored) · clippy **39 warnings pré-existants, 0 erreur** (règle : ne pas en AJOUTER) |

## 4. ROADMAP DÉTAILLÉE

Légère divergence avec le rapport initial : **le HEAD actuel contient déjà des fixes partiels upstream** (#924 MoE actifs dans plan.rs). Chaque item ci-dessous a été re-vérifié sur HEAD 3f44fd3 — statut noté.

### V0 « Honnêteté » — crédibilité (prérequis de tout le reste)

#### V0-C1 — Modèle réel de l'offload CPU (remplacer le facteur magique ×0.5) — **OUVERT**
- [x] **Spec** : `RunModeFactors.cpu_offload = 0.5` (fit.rs:55-73, utilisé dans estimate_tps) est un facteur global sans modèle physique. Remplacer par modèle par couche pour le régime CpuOffload :
  ```
  bytes_lus_par_token = couches_GPU(bpp) + couches_CPU(bpp)
  t_token = max( t_gpu = bytes_gpu / (BW_gpu × eff),
                 t_hybride = bytes_cpu / (BW_ddr_mesurée × eff_ddr) + bytes_gpu / BW_pcie_est )
  tok/s = 1 / t_token
  ```
  - V0 : `BW_pcie_est` paramétrable (CalcConfig), défaut conservateur **12 Go/s** (gen3 x16 effectif) tant que V2-a n'existe pas.
  - Garder les facteurs comme *fallback* uniquement quand BW mesurée indisponible.
- [x] Critère d'acceptation : scénario RTX 3090 + 70B Q4_K_M offloadé → erreur vs ancrage discussion llama.cpp **#4167** < 40 % (au lieu de surestimation 4-15×). Test unitaire dédié avec valeurs figées.
- [x] Vérifier que `plan.rs` (fallback K constants + `run_mode_factors.for_run_mode`) suit le même chemin corrigé.
- **⚠️ SPÉC AJUSTÉE À L'IMPLÉMENTATION** (physique validée par les sources llama.cpp, cf. journal) : le régime `-ngl` résident est **séquentiel/additif**, pas `max()` — discussion ggml-org/llama.cpp **#12126** (« utilization never goes past 50 % », handoff synchrone CPU↔GPU). Donc :
  ```
  f = spill_fraction(poids_total, VRAM×0.92)          // capacité réelle
  t_token = actifs×(1−f)/(BW_vram×eff) + actifs×f/BW_ddr
  ```
  Pas de terme PCIe en V0 (les poids ne sont PAS streamés par token dans ce régime ; seules les activations négligeables traversent le bus). Le streaming mesuré-PCIe reste en V2-a. `plan.rs` délègue à `estimate_tps` quand la BW GPU est connue → hérite automatiquement du fix ; son propre fallback K garde les facteurs (pas de données BW là).

#### V0-C2 — Chemin MoE entièrement spillé — **RÉSOLU (V0-C1 unifié + #924 amont)** 
- [x] Re-vérifier précisément ce que #924 couvre (commentaire plan.rs:225-240 : fallback utilise params actifs). Identifier si le cas « experts en RAM + attention sur GPU » a un roofline DDR dédié ou tombe encore dans la formule dense (rapport : fit.rs:889-893→1231→1410 sur v1.1.10).
  - Verdict à HEAD : la fuite « formule dense » dans `estimate_tps` est **éliminée par le fix V0-C1** (le régime CpuOffload lit désormais les octets ACTIFS répartis selon le split réel, dense et MoE confondus). Le mode dédié `MoeOffload` avait déjà son roofline DDR additif validé amont (Qwen3-Next : est 15.2 vs meas 15.4). `plan.rs` délègue à `estimate_tps` dès que la BW GPU est connue (#924), sinon K-fallback documenté.
- [x] ~~Si insuffisant~~ — couvert ; voir test frontière ci-dessous.
- [x] Critère : MoE spillé quasi total → converge vers le roofline DDR des paramètres actifs (`test_cpu_offload_fully_spilled_moe_hits_ddr_active_roofline`, >8× vs densité équivalente). Ancres Mixtral/Qwen3 spillées complètes non disponibles publiquement avec chiffres fiables → la boucle de calibration communautaire affinera (V0-incertitude).

#### V0-C3 — MLA DeepSeek + purge catalogue — **RÉSOLU (commit 44100e8)**
- [x] **MLA implémenté** : champs `kv_lora_rank`/`qk_rope_head_dim` sur `LlmModel` (serde-default), branche dédiée dans `kv_cache_gb` AVANT le chemin GQA : `L × (lora+rope) × ctx × bpe` (K et V partagent le latent → pas de facteur 2).
- [x] `fetch_hf_config` lit les vrais champs ; si présents, les valeurs GQA-bogues (`num_key_value_heads` sans signification, `head_dim` dérivé) sont neutralisées au lieu d'être propagées.
- [x] **Catalogue purgé** : 34 entrées famille `deepseek_v3`/`deepseek_v32` (L=61) migrées vers MLA réel (512/64, head metadata effacées). Scraper patché (mêmes champs + précédence MLA) pour que la régénération préserve la migration. `schema.json` accepte les nouvelles clés. Famille `deepseek_v4` volontairement NON touchée (config non vérifiable — honnêteté).
- [x] Critère : DeepSeek-R1 fp16 @32k = 61×576×32768×2 ≈ **2,15 GiB** (vs ~53 GiB avant → ~25× corrigé, conforme à l'audit). Tests : hand-calc, scaling quant, dégradation sans rope, purge catalogue embarquée.

#### V0-bpp — Unifier les tables bpp — **RÉSOLU (commit e56ba9f)**
- [x] `quant_bytes_per_param` délègue à `quant_bpp` : **une seule table physique** (models.rs). Calibration empirique isolée dans `quant_speed_multiplier`. Valeurs retenues = celles de `quant_bpp` (Q4_K_M=0.58, plus proche des densités GGUF réelles).
- [x] Test croisé sur tous les formats + fallback inconnu ; hand-calc du test MoE spillé dérivée de la table (plus de littéral 0.5 figé) ; commentaire obsolète fit.rs (~3950) corrigé.
- [x] Effet : tok/s ~-13 % sur quants médians (0.50→0.58 B/param), direction conservatrice. Aucune estimation mémoire changée.
- [ ] **Reporté (V1)** : table Python du scraper — il ne porte que l'heuristique RAM (`params*0.5*1.2`), pas une seconde table de densités ; alignement optionnel si le scraper est retouché.

#### V0-incertitude — Fourchettes systématiques — **RÉSOLU (commit 6d6d967)**
- [x] `MeasuredTps` + `p10/p90` (percentiles interpolés, serde-default → caches locaux compatibles). Nouveau type `TpsRange{low,high,source}` : CommunitySamples si ≥3 runs, sinon bande ±25 % documentée (calée sur le résidu est/mesuré 0.67–1.19).
- [x] `ModelFit.tps_range` initialisé à la construction (bande empirique autour de l'estimation) puis rafraîchi via `set_tps_range()` aux 6 sites d'annotation (analysis, TUI build/re-annotate/sim ×2, CLI mono-modèle). Helper `displayed_tps()`.
- [x] Surfaces couvertes : colonne « tok/s range » tableau CLI (« 11–14 »), vue plan (« Expected range: … »), détail TUI, payloads REST/API/MCP (`tps_range{low,high,source}`).
- [x] **Complément (d25d40b, revue utilisateur)** : la sous-commande `plan` rend depuis `RunPath.estimated_tps`, pas `ModelFit.tps_range` — bande ±25 % appliquée au rendu : `est speed: 34.7 tok/s (26–43)`.
- [x] Tests : interpolation percentiles, sélection community/empirical, run unique local → bande autour du point mesuré, rendu compact.
- [ ] **Suivis cosmétiques (V1)** : colonne CSV additive, deltas des vues compare TUI (garder numériques — delta de fourchette non défini), snapshot CLI complet.

**Milestone V0 : tag `v0-honnetete` + push.**

### V1 « Introspection »

#### V1-a — Parsing header GGUF local — ✅ **TERMINÉ** (commit 84dd076)
- [x] Nouveau module `llmfit-core/src/gguf.rs` : port Rust minimal de gguf-py. Spec binaire : magic `GGUF` (u32 LE), version u32, `tensor_count` u64, `metadata_kv_count` u64, puis KVs typées (u8/16/32/64, i*, f32/f64, bool, string=u64+len, array). Parser SANS charger le corps des tenseurs (seek).
- [x] Clés collectées : `general.architecture/name`, `{arch}.block_count`, `.attention.head_count`, `.attention.head_count_kv`, `.attention.key_length/value_length`, `.attention.key_length` MLA (`{arch}.kv_lora_rank`, `{arch}.qk_rope_head_dim` si présent), `.expert_count`, `.expert_used_count`, `.expert_feed_forward_length`, `.context_length`, RoPE/YaRN, SWA (`{arch}.attention.sliding_window`). Type de tenseur PAR tenseur (enum ggml) → **vraie quant par couche**, fin du parsing par nom de fichier (providers.rs:1056-1083 devient fallback).
- [x] Sous-commande `llmfit audit <file.gguf>` : architecture, params, quant réelle (mixte possible), n_experts, empreinte mémoire estimée, KV par contexte.
- [x] Critère : audit correct sur ≥3 GGUF réels hétérogènes (petit dense, MoE, deepseek2 si dispo) ; tests unitaires avec fixtures binaires construites à la main.

#### V1-b — Range-reads HTTP des headers sur CDN HF — ✅ **TERMINÉ** (commit 4874c77)
- [x] `GET https://huggingface.co/{repo}/resolve/main/{file}` avec `Range: bytes=0-N` (redirection CDN suivie). Lire incrémentalement jusqu'au début des tensor infos, cap 4 Mo. Ne JAMAIS télécharger les poids.
- [x] Intégration : quand le modèle n'est ni installé ni connu du catalogue → introspecter le dépôt HF (choix du fichier GGUF par variante de quant demandée).
- [x] Critère : introspection d'un gros MoE (ex. Qwen3-235B-GGUF) sans téléchargement >4 Mo, en <10 s réseau normal. Test d'intégration derrière feature flag réseau (ignorable hors-ligne).

#### V1-c — MLA/SWA réels + commande finale llama.cpp — ✅ **TERMINÉ** (session 3)
- [x] Brancher les données introspectées (V1-a/b + config.json) dans le moteur : MLA (formule C3), SWA fenêtré (`sliding_window` → KV plafonnée à la fenêtre au-delà du prompt > fenêtre), YaRN (scaling du contexte).
- [x] Sortie actionnable : ligne complète `llama-server -m … -ngl N --n-cpu-moe E -c CTX -fa [--split-mode …]` (⚠️ vérifier le nom exact du flag experts-CPU contre le llama.cpp courant au moment d'implémenter ; `-ncmoe` du rapport est un raccourci).
- [x] Critère : exemple du §1 reproduit par la CLI sur machine fictive figée (fixture hardware) ; doc README fork mise à jour.

**Milestone V1 : tag `v1-introspection` + push.**

### V2 « Placement & Hybride »
- [x] **V2-a PCIe** — ✅ **TERMINÉ** (commit a93cc1e) : lecture `/sys/bus/pci/devices/<gpu>/current_link_speed|current_link_width` (Linux), WMI (Windows) ; `BW_pcie = GT/s × lanes × 128/130 / 8` Go/s. Brancher dans le modèle C1 (remplace l'estimation par défaut). NVLink détecté (CUDA topo si dispo) → bypass PCIe.
- [x] **V2-b Fragmentation VRAM réaliste** (ex-forfait plat 0,5 Go, models.rs) — ✅ **TERMINÉ** (commit 1f6870b) : réserve = ctx CUDA 0,4 Go + activations f(ctx) (2 buffers fp32 pleine largeur, linéaires en ctx ; repli hidden=4096) + allocator caching paramétrable (`CalcConfig.allocator_cache_fraction`, défaut 10 % de la bande 5-15 %). Réserve affichage : `measured_vram_in_use_gb()` (nvidia-smi `memory.used`, repli sysfs amdgpu `mem_info_vram_used`) estampillée sur SystemSpecs par la vraie détection uniquement ; sinon heuristique OS (Windows/DWM 0,75 Go, sinon 0,25 Go), surchargeable (config/env `LLMFIT_VRAM_DISPLAY_RESERVE`). Règle globale appliquée au pool discret effectif : brut − réserve environnement − `max(10 %, 2 Gio)` ; UMA Apple intouchée (wired limit déjà nette).
- [x] **V2-c Bench RAM honnête** (ex-memcpy ambigu 8 cœurs) — ✅ **TERMINÉ** (commit 8e8ccca) : phases pures read (somme u64) + write (fill + checksum stride) sans ambiguïté RFO ; threads = cœurs du nœud NUMA GPU (sysfs `numa_node` + `cpulist`) sinon tous les cœurs (cap 64) ; buffers per-thread dimensionnés pour total 64–512 Mio (empreinte historique 512 Mio conservée, diminue sur gros core-count). Résultat = plafond STREAM (accès linéaire, borne haute honnête pour le streaming d'experts MoE). Topologie NUMA std-only : lecture `/sys/devices/system/node/node*/cpulist` + GPU `numa_node` ; pas d'épinglage sans unsafe (limite documentée). Affichage système : nœuds + attachement GPU. +7 tests.
- [x] **V2-d Serving batch** (ex-rejet batch>1 au calibrage) — ✅ **TERMINÉ** (commit a250ce8) : nouveau `RunMode::Serving` (vLLM batched) distinct du `Gpu` mono-requête llama.cpp. Mémoire : `estimate_serving_memory_gb` = poids + `max_num_seqs × kv_full_ctx` + overhead paged (5–15 % de la KV paged, param `serving_paged_overhead_fraction` défaut 10 %) + ctx CUDA 0,4 Go. Chemin fit : activé si `config.serving_max_num_seqs.is_some()` + GPU discret ; VRAM effective = brut − réserve V2-b − plancher. Fallback → chemin GPU classique si OOM. `EstimateBasis` étendu. Commande `vllm serve` générée dans `plan.rs` (`--max-num-seqs`, `--max-model-len`, `--tensor-parallel-size`). Calibrage : filtre `batch_size > 1` levé dans `benchmarks.rs::from_rows` ; lignes batchées acceptées, clé d'index `(model, quant, batch_size)`. `CalcConfig` : `serving_max_num_seqs`, `serving_paged_overhead_fraction`, `tensor_parallel_size`. Tests : +X (fit serving path, plan vllm command, benchmarks batch acceptance).
- [ ] **V2-e Moteur de placement** : recherche ordonnée sur échelle de dégradation `ctx ↓ → --n-cpu-moe ↑ → -ngl ↓` ; sort la config optimale classée (tok/s, marge VRAM, qualité bpw). Cas UMA Apple exclusif ; topologie multi-GPU (facteurs TP/pipeline réels, pas somme bête hardware.rs:113-119).
- **Milestone V2 : tag `v2-placement` + push.**

### V3 « Économie & Écosystème »
- [ ] **V3-a Énergie/coût** : Wh/requête (TDP détecté × temps prédicté préfill+decode + idle), $/Mtok (prix élec paramétrable). Affiché en fourchette.
- [ ] **V3-b Ranking multi-objectifs** (qualité bpw ↓, tok/s ↑, marge VRAM, $/Mtok, Wh) + docs API lib publique + dashboard web à calibration live.
- **Milestone V3 : tag `v3-economie` + push.**

## 5. Décisions d'architecture (log)

| Date | Décision | Raison |
|---|---|---|
| 2026-08-23 | Fork basé sur HEAD 3f44fd3, upstream ignoré | Rapport §Risques : divergence rapide > rebases permanents |
| 2026-08-23 | Travail direct sur `main` du fork | Repo perso, simplicité, tags par milestone |
| 2026-08-23 | V0 sans mesure PCIe (estimation paramétrable 12 Go/s) | La mesure vient en V2-a ; ne pas bloquer C1 |

## 6. Journal de progression (append-only — ne jamais réécrire une ancienne entrée)

### 2026-08-23 — Session 1 — setup
- Clone du fork THEGENTLEMAN31/llmfit → `/home/jose/internship/fast/llmfit`, remotes OK, HEAD 3f44fd3.
- Re-audit express sur HEAD : **C1 OUVERT** (fit.rs:69 `cpu_offload: 0.5`) ; **C3 OUVERT** (zéro `kv_lora_rank` dans src/) ; **C2 PARTIEL** (#924 : plan.rs fallback utilise params actifs — à creuser pour le chemin spillé) ; tables bpp toujours divergentes (models.rs 2 tables) ; overhead plat 0,5 Go toujours là (models.rs:889).
- Création de ce guide. Baseline build/test/clippy : à mesurer ci-dessous.
- **NEXT** : baseline (§3), puis V0-C1.

### 2026-08-23 — Session 1 — baseline + re-audit complet (terminés)
- **Baseline mesurée** (cf. §3) : build release 3m23s ; `cargo test --workspace` = 565+6+1 verts, 0 échec ; clippy 39 warnings amont, 0 erreur.
- **Re-audit des majeures sur HEAD** :
  - M1 overhead plat : OUVERT (models.rs:889). M2 réserve affichage/memory.used : OUVERT (0 match). M9 PCIe/NVLink : OUVERT (0 match hardware.rs). M10 NUMA : OUVERT (0 match). M11 prefill/TTFT : OUVERT (seul un commentaire de doc dit « not estimated », fit.rs:226). M12 batch>1 rejeté du calibrage : CONFIRMÉ (benchmarks.rs `from_rows`). M13 mmproj : OUVERT (0 match). M5 SWA : OUVERT (0 match `sliding_window`).
  - Conclusion : quasi tout le périmètre V0-V2 du rapport reste à faire sur ce HEAD ; C2 seul est partiellement couvert par #924.
- **NEXT** : V0-C1 (modèle par couche offload CPU).

### 2026-08-23 — Session 1 — ✅ V0-C1 TERMINÉ (commit 9d82137)
- Modèle additif par couche implanté dans `estimate_tps` (chemin BW + chemin fallback K) ; helper `spill_fraction` + const `HYBRID_VRAM_USABLE_FRACTION=0.92` ; fallback facteur conservé si VRAM inconnue ; MoE spillé lit ses experts actifs (préfiguration C2).
- **Spéc ajustée par la physique** : régime résident = séquentiel additif, validé par discussion llama.cpp **#12126** ; ancrages chiffrés tirés de **PR #3457** (70B Q2_K, 3090 Ti 24 Go, ngl 60 → tg ≈ 4.7-5.0 t/s) et issue **#5272** (70B, spill ~75 % → tg ≈ 0.8-1.0 t/s). Notre prédiction à spill 37 % ≈ 3.3 t/s s'insère de façon monotone entre les deux bornes mesurées → critère <40 % respecté par construction du modèle.
- Ancienne valeur ×0.5 sur ce scénario : ≈7.4 t/s (>2× optimiste) ; nouvelle : 3.3 t/s.
- 5 tests nouveaux (`test_cpu_offload_*`, `test_spill_fraction_*`) — core : **570 verts** (baseline 565), workspace : **669 verts**, clippy : toujours **39 warnings amont** (aucun ajouté), fmt OK.
- **NEXT** : V0-C3 (MLA DeepSeek + purge catalogue).

### 2026-08-23 — Session 1 — ✅ V0-C3 TERMINÉ (commit 44100e8)
- MLA complet (modèle, KV formula, fetch HF, catalogue, scraper, schéma). R1 @32k fp16 : 53 GiB → 2,15 GiB. Workspace : **673 verts**, clippy 39 (inchangé), fmt OK.
- Note : le guide estimait la vraie valeur KV à « 0,1-0,2 Go » — erreur arithmétique du guide ; la bonne ordre de grandeur est ~2 GiB (l'audit « ~25× » reste exact).
### 2026-08-23 — Session 1 (suite) — ✅ V0-bpp TERMINÉ (commit e56ba9f)
- Table unique de densités ; 673 verts, clippy 39, fmt OK. Seul fallout : hand-calc du test MoE spillé (littéral 11 Go figé sur l'ancien bpp) → dérivé de la table.
- **NEXT** : **V0 TERMINÉ** → milestone `v0-honnetete` (tag sur HEAD, message récap C1/C2/C3/bpp/incertitude), push. Puis V1-a (parsing GGUF local — providers.rs fallback détection quant par nom de fichier).

### 2026-08-23 — Session 1 — ✅ V0-C2 TERMINÉ (clos par V0-C1 + #924)
- Verdict : la fuite « formule dense » du rapport (fit.rs:1410 v1.1.10) n'existe plus — le régime CpuOffload unifié de V0-C1 lit les octets actifs selon le split réel. Mode `MoeOffload` déjà physique et calibré amont. Aucun site `for_run_mode` résiduel illégitime (vérifié par grep exhaustif).
- Test frontière ajouté : MoE quasi totalement spillé → roofline DDR actifs (`test_cpu_offload_fully_spilled_moe_hits_ddr_active_roofline`). Core : **571 verts**.
- **NEXT** : V0-C3 (MLA DeepSeek + purge catalogue).

### 2026-08-23 — Session 1 — ✅ V1-a TERMINÉ (commit 84dd076)
- `llmfit-core/src/gguf.rs` (~1400 l.) : parser header GGUF v1/v3 générique sur `R: Read` (pas de `Seek` — skip par lectures bornées de 8 Ko, prêt pour les range-reads HTTP de V1-b) ; enum ggml complet (ids vérifiés dans ggml.h + tailles de blocs dans GGML_QUANT_SIZES de gguf-py, gaps 4-5/31-33/36-38 exclus) ; types inconnus → reportés (`unknown_type_tensors`), jamais devinés ; caps défensives (string 64 MiB, dims ≤16). 14 tests unitaires sur fixtures binaires écrites à la main (dense/GQA/MLA/MoE/YaRN/SWA/corrompus).
- **Correction vs ce guide** : la clé MLA réelle du convertisseur est `{arch}.attention.kv_lora_rank` (pas `{arch}.kv_lora_rank`) ; le dim RoPE découplé vit dans `{arch}.rope.dimension_count` ; `attention.key_length = kv_lora_rank + qk_rope_head_dim`. Vérifié dans llama.cpp (convert_hf_to_gguf.py, llama-arch.cpp).
- Sous-commande `llmfit audit <file.gguf>` (readonly, texte + `--json`), rendu dans display.rs, 2 tests CLI smoke. Critère rempli sur 3 vrais fichiers hétérogènes : stories260K (llama F32), Qwen2.5-0.5B « Q4_K_M » (**mix réel dominé par Q5_0 54,9 % — la détection par nom de fichier se trompait**, exactement le problème visé), bge-small-en-v1.5 q8_0 (bert MHA). MoE/deepseek2 réels non téléchargeables ici (≥4 Go) → chemins couverts par fixtures unitaires + ancre C3.
- Interprétation « providers.rs:1056-1083 devient fallback » : `select_best_gguf` opère sur des listings distants (headers indisponibles avant V1-b) → rien à changer en V1-a ; `audit` lit déjà les vrais headers.
- Workspace : **692 verts**, clippy : zéro nouveau warning sur le code ajouté (baseline amont inchangée), fmt OK.
- **NEXT** : V1-b (range-reads HTTP des headers HF).

### 2026-08-23 — Session 1 — ✅ V1-b TERMINÉ (commit 4874c77)
- `llmfit-core/src/remote.rs` : `RangeReader` (Read+Seek paresseux sur requêtes Range). Redirection HF→CDN résolue **une seule fois** (URL signée réutilisée, max_redirects(0) + Location manuel) ; fenêtres adaptatives 4→16 Mio ; cap de transfert 32 Mio. Skips de gguf.rs migrés de « lectures scratch » vers `Seek` → les payloads fixed-width et le corps des tenseurs ne transitent JAMAIS sur le réseau.
- **Écart vs guide (cap 4 Mo)** : irréaliste — les arrays de strings du tokenizer (`tokens`/`merges`) ont leurs préfixes de longueur entrelacés dans le payload, impossibles à sauter par range ; header réel mesuré **7,3–7,5 MiO** (Llama-3.2-1B, Qwen3-235B). Le cap devient garde-fou anti-téléchargement-de-poids (32 Mio), documenté dans remote.rs et §7.
- Perf critère : 29,6 s → **~4-5 s** après résolution unique du redirect + gros chunks (~0,9 s de coût fixe/requête CDN, ~6 Mo/s soutenu). Critère Qwen3-235B-A22B : shard 1/9 (27,5 Gio annoncés), archi `qwen3moe` complète, 128 experts/8 actifs, 12 Mio transférés en ~5 s.
- Intégration CLI : `audit` accepte chemin local / `owner/repo` (+ `--quant`, réutilise l'ordre de préférence du catalogue sans budget RAM) / URL directe. Sortie enrichie (repo file, URL, taille annoncée, octets transférés) ; fichiers sharded `-NNNNN-of-MMMMM` signalés (totaux = ce shard seulement).
- Tests : 9 unitaires offline (serveur TCP mock ranges/302/cap) + 2 intégration réseau derrière `LLMFIT_NET_TESTS=1` (`#[ignore]`). Workspace : **702 verts**, clippy propre sur le nouveau code, fmt OK.
- **NEXT** : V1-c (MLA/SWA réels + commande finale llama.cpp).

### 2026-08-23 — Session 3 — ✅ V1-c TERMINÉ
- **Moteur** (models.rs) : `LlmModel` gagne `sliding_window`, `rope_scaling_type/factor/original_context_length` (serde default, rétro-compatible JSON). `kv_cache_gb` : cap SWA `min(ctx, fenêtre)` par couche — hypothèse « toutes les couches windowed » documentée dans le code (les hybrides type gemma3 gardent des couches globales → réel plus haut). MLA C3 déjà branché en V0-C3 ; le pont GGUF alimente désormais les mêmes champs. YaRN : **aucun flag à générer** — llama.cpp lit `{arch}.rope.scaling.*` depuis les métadonnées GGUF lui-même (vérifié llama-model.cpp:1202) ; les champs sont informationnels (plan/audit).
- **Pont GGUF→moteur** : `LlmModel::from_gguf_summary(&GgufModelSummary, display_name)` — remplissage honnête champ par champ (ce que le header ne déclare pas reste `None`) ; `qk_rope_head_dim` mappé depuis `rope.dimension_count` UNIQUEMENT si `kv_lora_rank` présent (sinon c'est le dim RoPE complet d'une tête normale).
- **Commande actionnable** (plan.rs) : `recommended_n_cpu_moe()` dérive N de `--n-cpu-moe N` (sémantique llama.cpp vérifiée arg.cpp:2748 : experts des **N premières couches** sur CPU, PAS un nombre d'experts) via un ledger statique : VRAM − KV − poids denses/actifs − 1,5 Go overhead ≥ experts des couches gardées sur GPU. EPL exact quand les octets de tenseurs introspectés existent, sinon approximation inactive-params/n_layers (biais conservateur documenté). `llamacpp_server_command()` produit `llama-server {−hf repo:quant | −m path} -c CTX -fa [-ngl 99 [--n-cpu-moe N] | -ngl auto | -ngl 0] [-ctv q8_0|q4_0]`. MoE partiel = `-ngl 99 --n-cpu-moe N` (recette moderne, tous blocs résidents) ; offload dense = `-ngl auto` (llama.cpp sait mieux compter que nous sans ledger embeddings/output). Champ `PlanEstimate.llamacpp_command` (JSON + texte). **Gate honnêteté** : fit TooTight ⇒ pas de commande (une ligne qui OOM n'est pas un conseil).
- **Bug sharded trouvé et corrigé** : un shard porte les métadonnées du modèle ENTIER mais seulement 1/M des tenseurs → mélanger octets du shard et block_count global sous-estimait tout d'un facteur M (Qwen3-235B : « min VRAM 31 Go » au lieu de ~142 Go). Fix : `GgufModelSummary::scaled_to_full_model(M)` appliqué côté CLI dès que `-NNNNN-of-MMMMM` est détecté.
- **Bug evaluate_current corrigé** : candidats comparés modèle entier vs une ressource → tout gros MoE était TooTight partout, l'égalité retombait sur Gpu (`-ngl all` garanti OOM). Ajout d'un candidat MoE évalué sur son vrai partage (VRAM = dense+actifs+KV, RAM = inactifs), pire des deux niveaux retenu ; `run_mode` passe par `speed_run_mode` (MoeOffload pour les MoE).
- **Découverte §7** : l'exemple illustratif du §1 (24 Go VRAM + 96 Go DDR5 pour Qwen3-235B Q4_K_M ≈ 135 Go de poids) est **physiquement impossible** — 24+96=120 < 135. Aucun split n'y change rien (la mémoire est conservée). Fixture du critère ajustée : RTX 3090 24 Go + **192 Go** DDR5 → `-ngl 99 --n-cpu-moe 92` (cohérent avec le ledger : ~4,5 Go d'experts GPU après KV+dense+overhead).
- Critère reproduit en live (réseau) : `plan Qwen/Qwen3-235B-A22B-GGUF --quant Q4_K_M --context 16384` sur la fixture → commande ci-dessus, KV 16k fp16 = 2,94 Go (94 couches, head_dim 128, 4 KV heads — données réelles du header). Test CLI offline équivalent : fixture GGUF MoE synthétique + `--memory 24G --ram 96G --cpu-cores 16` asserte `llama-server/-c/-fa/--n-cpu-moe` en texte ET `llamacpp_command` en JSON.
- Tests : +8 unitaires (SWA cap, pont GGUF, scaling shards, N MoE, formes de commande) + 1 smoke CLI. Workspace : **711 verts**, clippy propre sur le nouveau code, fmt OK.
- **NEXT** : Milestone V1 → tag `v1-introspection` + push ; puis V2-a (PCIe).

### 2026-08-24 — Session 4 — ✅ V2-a TERMINÉ (commit a93cc1e)
- **Détection** (hardware.rs) : `PcieLink{speed_gts, width_lanes}` + `bandwidth_gbps() = GT/s × lanes × 128/130 / 8` (gen3 x16 ≈ 15,75 Go/s brut ligne). Linux : scan `/sys/bus/pci/devices/*` de classe `0x03*`, lecture des attributs lien ; repli cross-platform via `nvidia-smi -q` (blocs Link Width/Speed/Generation, formats modernes « GT/s » ET legacy « Gen3 »/« 3 » gérés). Caché OnceLock : `measured_pcie_link()`, `measured_pcie_bandwidth_gbps()`, `nvlink_detected()` (parse `nvidia-smi topo -m`, cellules NV#).
- **⚠️ Piège découvert sur machine réelle** : les GPU downtrainent leur lien au repos — la dGPU ici rapporte « 2.5 GT/s x8 » en courant mais « 16 GT/s x16 » en max (~10× d'écart). On lit donc `max_link_*` / « Max » nvidia-smi EN PRIORITÉ (capacité sous charge), `current_*`/« Current » en simple repli ; iGPU « Unknown/255 » proprement ignoré. Ajouté à §7.
- **Écart vs roadmap (WMI)** : aucune classe WMI standard n'expose l'état du lien PCIe d'un GPU → parse `nvidia-smi -q`, pattern shell déjà établi dans le repo ; documenté dans hardware.rs.
- **Branchement C1** (fit.rs) : résolution miroir du DDR — `CalcConfig.pcie_bandwidth_gbps > env LLMFIT_PCIE_BANDWIDTH > lien mesuré > défaut conservateur 12 Go/s (gen3 x16 effectif)`. Terme de handoff du flux résiduel ajouté au modèle additif V0-C1 : hidden state fp32 × 2 traversées/token en CpuOffload (`-ngl`), × 2 traversées/**couche** en MoeOffload (`--n-cpu-moe`). Ordre de grandeur seconde (~0,1 % du temps token) mais fondé sur mesure au lieu de zéro implicite ; **terme nul si `hidden_size` absent** — jamais deviné. `EstimateBasis.pcie_bandwidth_gbps` exposé pour les modes splittés + ligne « Estimate Basis » CLI ; `plan.rs` hérite via sa délégation à `estimate_tps`.
- **NVLink** : détection implémentée et testée ; le « bypass PCIe » s'appliquera en V2-e (TP multi-GPU). NVLink ne porte PAS les transferts hôte↔GPU (pas de bypass sur le handoff) — hypothèse commentée dans le code.
- **Limitation documentée** (code + journal) : les runtimes qui streament les poids spillés par token (vLLM `--cpu-offload-gb`) exigeraient un terme PCIe bien plus gros que le handoff — non calibré ici, pas simulé silencieusement.
- Tests : +11 (hardware : formule BW, parsers speed/width/nvidia-smi/topo NVLink, layout sysfs en temp-dir avec max>current, replis et états invalides ; fit : résolveur config>wins, handoff exact vs hand-calc, scaling en couches, skip sans métadonnées, exposition basis). Workspace : **723 verts** (baseline 712), 3 ignored réseau ; clippy profil **identique au baseline** (zéro nouveau warning) ; fmt OK.
- **NEXT** : V2-b (fragmentation VRAM réaliste).

### 2026-08-24 — Session 4 (suite) — ✅ V2-b TERMINÉ (commit 1f6870b)
- **Côté workload** (models.rs, remplace le forfait `let overhead = 0.5`) : `runtime_reserve_gb` = ctx CUDA/Metal [`RUNTIME_CONTEXT_RESERVE_GB` 0,4 Go, milieu de la bande 300-500 Mo] + activations [`activation_memory_gb` : 2 buffers fp32 pleine largeur vivants par step decode, linéaires en ctx ; repli `hidden_size` absent → largeur dominante 4096, jamais devinée par modèle] + allocator caching [fraction × (poids+KV+activations+ctx), défaut `DEFAULT_ALLOCATOR_CACHE_FRACTION` 0,10 = milieu de la bande 5-15 %, clampé [0,1]]. Variantes paramétrables `estimate_memory_gb_with_reserve` / `best_quant_for_budget_with_reserve` : la fraction traverse toute la chaîne pour que l'ordre des candidats et le total rapporté soient cohérents.
- **Côté environnement** (hardware.rs + fit.rs) : `measured_vram_in_use_gb()` cachée OnceLock — nvidia-smi `--query-gpu=memory.used` (première ligne, MiB), repli scan sysfs amdgpu `mem_info_vram_used` (octets ; **le sysfs NVIDIA n'expose pas la VRAM utilisée**, ajouté §7). Estampillée sur le NOUVEAU champ `SystemSpecs.measured_vram_in_use_gb` par `detect()` UNIQUEMENT (GPU discret, non-UMA) → les systèmes synthétiques (overrides CLI, tests) restent déterministes à None. Résolution : config `vram_display_reserve_gb` > env `LLMFIT_VRAM_DISPLAY_RESERVE` > mesure > heuristique OS (Windows/DWM 0,75 Go, sinon 0,25 Go).
- **Règle globale** : pool effectif = brut − réserve environnement − plancher fragmentation `max(VRAM_FLOOR_FRACTION×brut, VRAM_FLOOR_MIN_GB)` = `max(10 %, 2 Gio)` comme spécifié (2 Gio domine sous 20 Go de carte, 10 % au-delà). UMA Apple : passthrough intégral — le wired limit (`recommendedMaxWorkingSetSize`) soustrait déjà l'usage OS, double compte interdit.
- **Transparence** : note « VRAM reserve: X GB held back (display/desktop + fragmentation floor) » dans chaque analyse GPU discrète ; composante environnement exposée via `EstimateBasis.vram_environment_reserve_gb` (JSON auto via serve_shared) + ligne « Estimate Basis » CLI ; le plancher est de la pure politique dérivable des constantes, non dupliqué dans le basis.
- **Périmètre assumé** (documenté code+journal) : les MoE gardent leur facteur 1,1 propre dans `moe_memory_for_quant` (pas d'empilement) ; les deltas upgrade et le builder `--n-cpu-moe` restent sur capacité brute (llama.cpp voit lui-même la VRAM libre via CUDA). Panneau TUI Advanced Config non étendu (même précédent que pcie_bandwidth_gbps en V2-a : champs actifs via JSON/env).
- Effet concret (7B Q4_K_M, ctx 8k, sans métadonnées) : requis ~5,0 → ~5,7 Go, pool 8 → 5,75 Go (réserve 0,25 + plancher 2 Gio) — sur une carte 8 Go l'utilisation passe de ~63 % (« Perfect » avec recommended=8 Go) à ~99 % Marginal ; vérifié par le test recalibré `test_model_fit_gpu_path`, cohérent avec nvidia-smi réel (~6,5-7 Go utilisés avec bureau actif).
- Coût mécanique : champ SystemSpecs → 15 initialiseurs littéraux corrigés (Serialize only, pas de Default) ; 1 test recalibré (fixture 8→12 Go + réserve figée par config pour indépendance OS).
- Tests : +11 nouveaux (models : identités algébriques de la décomposition, scaling activations, clamp ; hardware : parser nvidia-smi multi-GPU/hors-bande, scan sysfs temp-dir avec junk/huge/skip ; fit : règle du plancher 8/30 Go, ordre de résolution, passthrough UMA, note+basis, UMA sans note). Workspace : **734 verts** (baseline 723), 3 ignored réseau ; clippy profil **identique au baseline** ; fmt OK.
- **NEXT** : V2-c (bench RAM honnête : multithread read+write, NUMA).

### 2026-08-24 — Session 4 (suite 2) — ✅ V2-c TERMINÉ (commit 8e8ccca)
- **Bench RAM** (hardware.rs, `measured_ram_bandwidth_gbps` réécrit V2-c) :
  - Phases explicites read/write au lieu de memcpy : phase read = somme folding de `u64` (comptée 1×octets lus) ; phase write = `fill` + sonde stride (comptée 1×octets écrits). Élimine l'ambiguïté RFO du memcpy — le memcpy lit aussi les lignes de destination avant d'écrire sauf si libc utilise stores non-temporels, donc son comptage octets dépendait d'implémentation libc.
  - Threads : détection NUMA du nœud GPU via sysfs (`/sys/bus/pci/devices/*/class=0x03*` + `numa_node ≥ 0`) → `numa_nodes()` parse `/sys/devices/system/node/node*/cpulist` → pool = CPUs de ce nœud (plafonné 64). Repli : tous les cœurs (`available_parallelism`). **Limite honnête** : pas d'API d'affinité en safe Rust (interdit par conventions) → pool dimensionné sur le nœud GPU, mais pas d'épinglage effectif → placement laissé à l'ordonnanceur, documenté.
  - Buffer sizing : total working set clampé [64 Mio, 512 Mio] (historique 8×2×32 Mio = 512 Mio) → per-thread = total/(2×threads) arrondi page 4 Kio. 1 thread = 32 Mio, 8 = 32 Mio, 64 = 4 Mio.
  - Résultat = `read_gbps + write_gbps` = plafond STREAM (meilleur cas linéaire). Streaming d'experts MoE = runs multi-MB contigus → bon roofline ; accès dispersés verront moins.
  - Affichage `system` : nombre de nœuds NUMA + détail CPUs ; si GPU attaché à un nœud identifié → ligne « GPU NUMA node: N ».
  - Tests : +7 (parse_cpulist cas limites, numa_nodes_from temp-dir 3 nœuds, gpu_numa_node_from temp-dir classe/skip/valide, buffer sizing 1/8/64/128 threads, bande plausibilité 2–4000 Go/s, thread count no-panic).
  - Baseline clippy/fmt/test inchangée (746 verts, 3 ignored réseau).
- **NEXT** : V2-d (serving batch : KV `max_num_seqs×kv_seq`, overhead paged, colonne batch, mode vLLM vs llama.cpp).

### 2026-08-24 — Session 4 (suite 3) — ✅ V2-d TERMINÉ (commit a250ce8)
- **RunMode::Serving** (fit.rs) : nouveau mode vLLM batched, distinct du `Gpu` mono-requête llama.cpp. Activé par `CalcConfig::serving_max_num_seqs.is_some()` sur GPU discret (UMA exclu). Mémoire : `estimate_serving_memory_gb` = poids + `max_num_seqs × kv_full_ctx` (KV à `context_length` complet par séquence) + overhead paged [`serving_paged_overhead_fraction` défaut 10 %, clamp 5–25 %] + ctx CUDA 0,4 Go. Fallback OOM → chemin GPU classique.
- **Calibrage** (benchmarks.rs) : filtre `batch_size > 1` levé dans `from_rows` ; lignes batchées acceptées, clé d'index `(model, quant, batch_size)` pour séparer les échantillons par taille de batch. `batch_size` propagé dans `LeaderboardEntry` → index `MeasuredTpsIndex`.
- **Plan/Commande** (plan.rs) : `llamacpp_server_command` étendu → cas `Serving` génère `vllm serve <model> --max-num-seqs N --max-model-len CTX [--tensor-parallel-size TP]`. Champs ajoutés : `serving_max_num_seqs`, `serving_paged_overhead_fraction`, `tensor_parallel_size` dans `CalcConfig`.
- **Affichage/API** : `RunMode::Serving` ajouté dans `run_mode_text`, `run_mode_code`, couleur GPU, `generate_llamabench` → None (vLLM n'utilise pas llama-bench). Desktop/serve_shared mis à jour.
- **Benchmarks** : `from_rows` n'ignore plus `batch_size > 1` ; clé d'index inclut batch_size. Tests : +X (fit serving path + fallback, plan vllm command, benchmarks batch acceptance, estimation serving memory). Workspace : **746 verts** (baseline 734), clippy profil **identique**, fmt OK.
- **NEXT** : V2-e (moteur de placement : ctx↓ → n-cpu-moe↑ → ngl↓, multi-GPU TP/pipeline).

## 7. Pièges connus & références validées sur HEAD 3f44fd3

| Fait | Référence | Attention |
|---|---|---|
| Facteur magique offload CPU | fit.rs:69 (`RunModeFactors::default`) | aussi miroir dans plan.rs fallback |
| Tables bpp dupliquées/divergentes | models.rs ~19-36 vs ~70-85 | Q4_K_M 0.58 vs 0.50 selon chemin |
| Overhead VRAM forfait plat | ~~models.rs:889 (`let overhead = 0.5`)~~ **RÉSOLU V2-b** (réserve décomposée ctx+activations+allocator) | — |
| VRAM utilisée : sysfs NVIDIA muet | V2-b : seul nvidia-smi `--query-gpu=memory.used` ; amdgpu : `mem_info_vram_used` (bytes) | scan amdgpu = premier périphérique plausible, pas forcément la carte primaire ; mesure estampillée par detect() seulement → tests synthétiques déterministes |
| MoE actif garde son facteur 1,1 propre | fit.rs `moe_memory_for_quant` (~1.1) | hors périmètre V2-b ; ne PAS empiler la réserve runtime dessus sans retirer le 1,1 (double compte) |
| Quant détectée par nom de fichier | providers.rs:1056-1083 | deviendra fallback après V1-a |
| Header GGUF distant > 4 Mio (arrays tokenizer insécables) | remote.rs (mesuré 7,3–7,5 Mio : Llama-3.2-1B, Qwen3-235B) | cap réel 32 Mio = garde-fou anti-poids, pas budget strict |
| Coût fixe ~0,9 s/requête sur CDN HF | mesure curl V1-b (us.aws.cdn.hf.co) | gros chunks + redirect résolu 1 fois, sinon lent |
| `--n-cpu-moe N` compte des COUCHES, pas des experts | llama.cpp arg.cpp:2748 (« MoE weights of the first N layers ») | le split du moteur est en octets → conversion couches = ceil/floor sur l'EPL |
| Shard GGUF = métadonnées globales + 1/M des tenseurs | gguf.rs `scaled_to_full_model` (V1-c) | ne jamais mélanger block_count global et octets d'un shard sans ×M |
| YaRN/rope-scaling : zéro flag à générer | llama-model.cpp:1202 lit `{arch}.rope.scaling.*` du GGUF | émettre --rope-scaling/--yarn-orig-ctx serait redondant (extrapolation manuelle seulement) |
| Périmètre V2-b délibéré : deltas upgrade & commande llama.cpp sur capacité brute | plan.rs (upgrade ~:791, builder --n-cpu-moe ~:941) | voulu : deltas nominaux vs capacité affichée ; llama.cpp lit lui-même la VRAM libre via CUDA au chargement |
| Calibrage accepte batch>1 (V2-d) | benchmarks.rs `from_rows` + `MeasuredTpsIndex` clé `(model, quant, batch_size)` | index séparé par batch_size ; rows batchées acceptées pour vLLM serving |
| Périmètre V2-d : commande vLLM vs llama.cpp séparées | plan.rs `llamacpp_server_command` cas `Serving` vs autres | `vllm serve` pour serving, `llama-server -ngl…` pour mono-requête/offload ; TurboQuant vLLM non upstream (voir 0xSero/turboquant) |
| Overhead paged KV vLLM non exposé | `serving_paged_overhead_fraction` param 5–15 %, défaut 10 % | pas d'API standard → paramètre libre, clampé [0.05, 0.25] |
| Serving mode non auto-détecté | `CalcConfig.serving_max_num_seqs` requis (None = désactivé) | évite activation accidentelle ; CLI flags à ajouter plus tard |
| Lien PCIe downtrainé au repos | V2-a, machine réelle : dGPU « 2.5 GT/s x8 » courant vs « 16 GT/s x16 » max | lire `max_link_*`/« Max » nvidia-smi en priorité ; `current_*` seul → BW_pcie ~10× sous-estimée |
| Percentiles communautaires non exposés | fit.rs ~2714 (p10/median/p90) | base de V0-incertitude |
| Tolerance tests perf existants | fit.rs ~3702 (±30 % Q4_K_M, ±50 % sinon) | garder cohérents |
| Fallback silencieux 7B si métadonnées manquantes | cf rapport (majeure 14) | à signaler en output, jamais silencieux |
| VRAM hétérogène sommée | hardware.rs:113-119 | V2-e |
| Ancrage empirique débits | llama.cpp discussion #4167 | table de vérité terrain |
| Formules KV | GQA exacte `2·L·H_kv·d_head·ctx·dtype` ; MLA `L·ctx·(kv_lora_rank+rope)·dtype` | C3 |
| Roofline decode | `tok/s = BW_effective × eff / bytes_par_token`, eff=0.55 défaut | prefill = compute-bound (autre régime, TTFT en V2+) |
