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

#### V0-C3 — MLA DeepSeek + purge catalogue — **OUVERT**
- [ ] **MLA absent** (zéro match `kv_lora_rank` sur tout src/ à HEAD). Formule correcte :
  `KV_MLA = L × ctx × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes` (par token : `(kv_lora_rank + rope)` octets×dtype/couche, PAS `2·H_kv·d_head`).
- [ ] Détection : champs `config.json` HF `kv_lora_rank` / `qk_rope_head_dim` (modèles deepseek2/deepseek-v3). Ajouter au struct modèle + branchement dans le calcul KV (models.rs calcule aujourd'hui `2·L·H_kv·d_head·ctx·dtype` partout).
- [ ] **Purger le catalogue** `data/hf_models.json` : entrées DeepSeek avec métadonnées inventées (`H_kv=128, d_head=56`) → KV surestimé ~25×. Script de validation croisée contre config.json HF live (comme `fetch_hf_config` update.rs:428-462), corriger/regénérer les entrées deepseek*.
- [ ] Critère : test unitaire KV DeepSeek-R1 @32k ctx ≈ valeur documentée (ordre de grandeur ~0,1-0,2 Go fp8, vs ~2,5+ Go avec l'erreur actuelle) ; `cargo test` vert.

#### V0-bpp — Unifier les tables bpp — **OUVERT**
- [ ] Deux tables divergentes : models.rs:~19-36 (`quant_bpp` défaut 0.58) vs models.rs:~70-85 (défaut 0.50) ; Q4_K_M = 0.58 vs 0.50 selon le chemin. Le scraper Python (scripts/) a sa propre table.
- [ ] Créer UNE source unique (module `quants.rs` ou constante sérialisée partagée `data/quants.json` générée), consommée par Rust + Python. Table de référence : tailles réelles GGUF (gguf-py / wiki llama.cpp).
- [ ] Critère : un seul endroit définit bpp ; tests croisés Rust/Python verts ; grep prouve l'absence de seconde table.

#### V0-incertitude — Fourchettes systématiques — **OUVERT**
- [ ] Aucune sortie ponctuelle de tok/s ne reste sans intervalle. Base : les percentiles communautaires existent déjà (fit.rs:~2700-2745 : p10/median/p90 calculés mais non exposés).
- [ ] Exposer `range` (p10-p90 quand ≥N échantillons communautaires pour le slug matériel, sinon ±25 % documenté) dans : CLI fit/plan, API REST, lib.
- [ ] Format « 11–14 tok/s ». Jamais de valeur seule.
- [ ] Critère : toute sortie débit des sous-commandes affiche une fourchette ; test snapshot.

**Milestone V0 : tag `v0-honnetete` + push.**

### V1 « Introspection »

#### V1-a — Parsing header GGUF local — **ABSENT**
- [ ] Nouveau module `llmfit-core/src/gguf.rs` : port Rust minimal de gguf-py. Spec binaire : magic `GGUF` (u32 LE), version u32, `tensor_count` u64, `metadata_kv_count` u64, puis KVs typées (u8/16/32/64, i*, f32/f64, bool, string=u64+len, array). Parser SANS charger le corps des tenseurs (seek).
- [ ] Clés collectées : `general.architecture/name`, `{arch}.block_count`, `.attention.head_count`, `.attention.head_count_kv`, `.attention.key_length/value_length`, `.attention.key_length` MLA (`{arch}.kv_lora_rank`, `{arch}.qk_rope_head_dim` si présent), `.expert_count`, `.expert_used_count`, `.expert_feed_forward_length`, `.context_length`, RoPE/YaRN, SWA (`{arch}.attention.sliding_window`). Type de tenseur PAR tenseur (enum ggml) → **vraie quant par couche**, fin du parsing par nom de fichier (providers.rs:1056-1083 devient fallback).
- [ ] Sous-commande `llmfit audit <file.gguf>` : architecture, params, quant réelle (mixte possible), n_experts, empreinte mémoire estimée, KV par contexte.
- [ ] Critère : audit correct sur ≥3 GGUF réels hétérogènes (petit dense, MoE, deepseek2 si dispo) ; tests unitaires avec fixtures binaires construites à la main.

#### V1-b — Range-reads HTTP des headers sur CDN HF — **ABSENT**
- [ ] `GET https://huggingface.co/{repo}/resolve/main/{file}` avec `Range: bytes=0-N` (redirection CDN suivie). Lire incrémentalement jusqu'au début des tensor infos, cap 4 Mo. Ne JAMAIS télécharger les poids.
- [] Intégration : quand le modèle n'est ni installé ni connu du catalogue → introspecter le dépôt HF (choix du fichier GGUF par variante de quant demandée).
- [ ] Critère : introspection d'un gros MoE (ex. Qwen3-235B-GGUF) sans téléchargement >4 Mo, en <10 s réseau normal. Test d'intégration derrière feature flag réseau (ignorable hors-ligne).

#### V1-c — MLA/SWA réels + commande finale llama.cpp — **ABSENT**
- [ ] Brancher les données introspectées (V1-a/b + config.json) dans le moteur : MLA (formule C3), SWA fenêtré (`sliding_window` → KV plafonnée à la fenêtre au-delà du prompt > fenêtre), YaRN (scaling du contexte).
- [ ] Sortie actionnable : ligne complète `llama-server -m … -ngl N --n-cpu-moe E -c CTX -fa [--split-mode …]` (⚠️ vérifier le nom exact du flag experts-CPU contre le llama.cpp courant au moment d'implémenter ; `-ncmoe` du rapport est un raccourci).
- [ ] Critère : exemple du §1 reproduit par la CLI sur machine fictive figée (fixture hardware) ; doc README fork mise à jour.

**Milestone V1 : tag `v1-introspection` + push.**

### V2 « Placement & Hybride »
- [ ] **V2-a PCIe** : lecture `/sys/bus/pci/devices/<gpu>/current_link_speed|current_link_width` (Linux), WMI (Windows) ; `BW_pcie = GT/s × lanes × 128/130 / 8` Go/s. Brancher dans le modèle C1 (remplace l'estimation par défaut). NVLink détecté (CUDA topo si dispo) → bypass PCIe.
- [ ] **V2-b Fragmentation VRAM réaliste** (aujourd'hui forfait plat 0,5 Go, models.rs:888-890) : réserve = ctx CUDA ~300-500 Mo + activations f(ctx,batch) + allocator caching 5-15 % paramétrable + réserve affichage (Windows/DWM 0,5-1 Go ; lire `memory.used` NVML/sysfs quand possible). Règle : réserve globale `max(10 %, 2 Gio)`.
- [ ] **V2-c Bench RAM honnête** : multithread read+write (le bench actuel est memcpy lecture-seule mono-pattern), awareness NUMA (/sys/devices/system/node, premier nœud du socket GPU si identifiable).
- [ ] **V2-d Serving batch** : KV dimensionnée `max_num_seqs × kv_seq` + overhead paged (~5-15 %) ; mode vLLM séparé du mode llama.cpp mono-requête ; lever le rejet batchSize>1 du calibrage (benchmarks.rs:452-464) avec colonne batch.
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
- **NEXT** : V0-C2 (chemin MoE spillé — vérifier ce que #924 laisse passer).

### 2026-08-23 — Session 1 — ✅ V0-C2 TERMINÉ (clos par V0-C1 + #924)
- Verdict : la fuite « formule dense » du rapport (fit.rs:1410 v1.1.10) n'existe plus — le régime CpuOffload unifié de V0-C1 lit les octets actifs selon le split réel. Mode `MoeOffload` déjà physique et calibré amont. Aucun site `for_run_mode` résiduel illégitime (vérifié par grep exhaustif).
- Test frontière ajouté : MoE quasi totalement spillé → roofline DDR actifs (`test_cpu_offload_fully_spilled_moe_hits_ddr_active_roofline`). Core : **571 verts**.
- **NEXT** : V0-C3 (MLA DeepSeek + purge catalogue).

## 7. Pièges connus & références validées sur HEAD 3f44fd3

| Fait | Référence | Attention |
|---|---|---|
| Facteur magique offload CPU | fit.rs:69 (`RunModeFactors::default`) | aussi miroir dans plan.rs fallback |
| Tables bpp dupliquées/divergentes | models.rs ~19-36 vs ~70-85 | Q4_K_M 0.58 vs 0.50 selon chemin |
| Overhead VRAM forfait plat | models.rs:889 (`let overhead = 0.5`) | pas de ctx CUDA/allocator/affichage |
| Quant détectée par nom de fichier | providers.rs:1056-1083 | deviendra fallback après V1-a |
| Calibrage rejette batch>1 | benchmarks.rs:452-464 | lever en V2-d |
| Percentiles communautaires non exposés | fit.rs ~2714 (p10/median/p90) | base de V0-incertitude |
| Tolerance tests perf existants | fit.rs ~3702 (±30 % Q4_K_M, ±50 % sinon) | garder cohérents |
| Fallback silencieux 7B si métadonnées manquantes | cf rapport (majeure 14) | à signaler en output, jamais silencieux |
| VRAM hétérogène sommée | hardware.rs:113-119 | V2-e |
| Ancrage empirique débits | llama.cpp discussion #4167 | table de vérité terrain |
| Formules KV | GQA exacte `2·L·H_kv·d_head·ctx·dtype` ; MLA `L·ctx·(kv_lora_rank+rope)·dtype` | C3 |
| Roofline decode | `tok/s = BW_effective × eff / bytes_par_token`, eff=0.55 défaut | prefill = compute-bound (autre régime, TTFT en V2+) |
