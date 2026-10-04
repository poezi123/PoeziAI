# Weniger Compute, gleiche Intelligenz — Forschungslektüre

> Stand der Forschung 2024–2026 zum Betreiben großer, „schlauer" Modelle auf
> schwacher Hardware, **ohne** die Intelligenz zu opfern. Zusammengestellt für
> OVERFLOW / PoeziAI. Datum: 2026-10-04.

Diese Lektüre ist der Landkarten-Teil unserer eigenen Forschung: *was es schon
gibt*, *wie gut es zu unseren Randbedingungen passt*, und *wo echte Lücken
bleiben, in denen wir selbst etwas beitragen könnten*. Die Techniken sind mit
Quellen belegt und jeweils ehrlich gegen unsere Situation bewertet — nicht jede
Technik aus einem A100-Paper funktioniert auf einer 3060 ohne eigene Kernels.

---

## 0. Unsere Randbedingungen (der Filter für alles Folgende)

| Faktor | Bei uns |
|---|---|
| Modell | **dichtes** Qwen2.5-Coder (7B, perspektivisch 14B/32B), SwiGLU/SiLU, kein MoE |
| GPU | RTX 3060, 12 GB (≈10 GB nutzbar mit Desktop) |
| CPU | ~37× langsamer **pro Token** als die GPU (≈0.7 vs ≈26 tok/s beim 7B) |
| Präzision | **voll** — keine Quantisierung (würde das Modell dümmer machen) |
| Umsetzung | **reiner Rust**, **keine eigenen CUDA-Kernels** mehr (die bestehenden bleiben) |
| Workload | agentisch, eng: Security + Code (kein Generalist) |

Zwei dieser Zeilen sind ungewöhnlich streng und schließen viel Literatur aus:
**(a) keine Quantisierung** und **(b) keine eigenen Kernels**. Fast alle
veröffentlichten GPU-Speedups brauchen maßgeschneiderte CUDA-Kernels (sparse
Gather, Huffman-Decode, fused MoE). Reiner Rust lässt uns v. a. auf der
**Orchestrierungs-Ebene** (Scheduling, Offloading, Decoding-Strategie, Caches)
und auf dem **CPU-Pfad** forschen — das ist beim Bewerten jeder Technik der
entscheidende Filter.

**Das „unmögliche Dreieck":** von {passt in VRAM} + {volle Qualität} + {volle
Geschwindigkeit} kann man bei einem dichten Modell, das größer als der VRAM ist,
nur zwei gleichzeitig haben — *solange man jedes Gewicht bei jedem Token liest*.
Die ganze Forschung unten ist der Versuch, genau diese letzte Annahme zu brechen.

### Was wir selbst schon gemessen haben (eigene Daten)
- **FFN = ~87 %** der Block-Gewichte → der FFN-Teil ist der Haupthebel.
- **Aktivierungs-Sparsity ist real:** nur ~4.5 % der FFN-Neuronen überschreiten
  5 % des Maximums pro Token, ~35 % überschreiten 1 %.
- **Oracle-Masking (`poezi masktest`):** die oberen **50 %** der Neuronen zu
  behalten ist **bit-identisch**; darunter sinkt die Top-1-Übereinstimmung sanft
  (25 % → 90 %, 10 % → 78 %).
- **Statischer Skew ist schwach (`poezi skew`):** die 25 % häufigsten Neuronen
  decken nur ~39 % der Aktivierungen ab (SiLU liefert kein Power-Law).

Diese Messungen sind, wie sich zeigt, **genau konsistent mit der Literatur**: ein
fester Hot-Satz (DejaVu/PowerInfer) wirkt nur bei ReLU-Modellen; bei SwiGLU muss
man **magnitude-basiert pro Token** vorgehen (TEAL/CATS) — das deckt sich mit
unserem „50 % verlustfrei, aber kein statischer Hot-Satz".

---

## 1. Kategorie A — Verlustfreie Gewichts-Kompression *(Modell passt, 0 Qualitätsverlust)*

Die direkteste Antwort auf „15 GB in 12 GB ohne Verlust": das Modell **verlustfrei
kleiner speichern** und beim Rechnen entpacken. Keine Quantisierung — die Ausgabe
bleibt bit-identisch.

- **DFloat11** (NeurIPS 2025) — die wichtigste Technik für unser Beispiel.
  BF16-Gewichte haben niedrige Entropie in den Exponenten; Huffman-Codierung der
  Exponenten ergibt **~30 % kleiner bei bit-identischer Ausgabe**. Damit lief ein
  810-GB-Modell auf 8×80 GB, das vorher nicht passte. Für uns: 15 GB → ~10.5 GB →
  **passt in die 3060**.
  → Paper: <https://arxiv.org/pdf/2504.11651> · Code:
  <https://github.com/LeanModels/DFloat11>
- **„Approaching Shannon Bound with Lossless LLM Weight Compression"** (2026) —
  treibt die verlustfreie Kompression näher ans informationstheoretische Optimum.
  → <https://arxiv.org/pdf/2606.15789>
- **NeuZip** — frühere verlustfreie Kompression (Vergleichsbasis von DFloat11).

**Verdict für uns:** ⚠️ **Konzeptionell perfekt**, praktisch mit zwei Haken.
(1) Das Entpacken braucht einen **GPU-Decode-Kernel** (Huffman) → kollidiert mit
„keine Kernels". In reinem Rust ginge CPU-Decode beim Laden in den VRAM, aber dann
sind die Gewichte dort wieder voll groß — der Fit-Gewinn ist weg. (2) DFloat11
zielt auf **BF16**; unsere Gewichte sind bereits **Q8_0** (quantisiert) — die
Entropie-Reserve ist kleiner. **Merke:** wenn wir ein *unquantisiertes* großes
Modell fahren wollen, ist DFloat11-artige verlustfreie Kompression der sauberste
Weg zum Fit — kostet uns aber die „keine Kernels"-Regel. Ein ehrlicher
Abwägungspunkt für dich.

---

## 2. Kategorie B — Aktivierungs-Sparsity *(weniger Rechnung pro Token, SwiGLU-tauglich)*

Die Erkenntnis, die unsere eigenen Messungen stützt: man muss nicht jedes
FFN-Neuron rechnen, weil die meisten pro Token ~0 beitragen. Der Trick ist, das
bei **SwiGLU/SiLU** (keine harten Nullen) hinzubekommen.

- **TEAL — „Training-Free Activation Sparsity in LLMs"** (ICLR 2025). Magnitude-
  basierte Sparsity auf den Hidden-States **im ganzen Modell**, ohne Retraining:
  **40–50 % modellweite Sparsity bei minimalem Qualitätsverlust**, 1.5–1.8×
  Wall-Clock-Speedup. Validiert auf Llama-2/3 & Mistral (7B–70B).
  → <https://arxiv.org/pdf/2408.14690> · ICLR:
  <https://iclr.cc/virtual/2025/poster/28994>
  **Das ist exakt unser Masktest-Ergebnis** (50 % behalten ≈ verlustfrei), nur als
  publizierte, breit validierte Methode.
- **CATS** — training-free Sparsity speziell für **SwiGLU**: Magnitude-Pruning auf
  dem **Wgate-Ausgang** (genau die Stelle, die wir messen).
- **SparseInfer** — **training-free Vorhersage** des aktiven Satzes, d. h. ein
  Prädiktor *ohne* Training. → <https://arxiv.org/pdf/2411.12692>
- **„First Activations Matter"** — training-free dynamische Aktivierung.
  → <https://arxiv.org/pdf/2408.11393>
- **Polar Sparsity** (2025) — verschiebt den Sparsity-Fokus mit Batch-Größe,
  Hochdurchsatz-Variante. → <https://arxiv.org/pdf/2505.14884>
- Kontext: **DejaVu/PowerInfer** nutzen einen *festen* Hot-Satz — funktioniert nur
  bei **ReLU**-Modellen (Power-Law). Bei SwiGLU **nicht** → deckt sich mit unserem
  schwachen Skew-Ergebnis.

**Verdict für uns:** ✅ **Qualitativ bestätigt, aber der Speed-Gewinn hängt am
Kernel.** TEAL/CATS zeigen, dass wir ~40–50 % der FFN-Rechnung streichen können,
ohne Qualität zu verlieren. **Aber:** Rechnung zu *sparen* heißt, nur die aktiven
Neuron-Zeilen zu laden/multiplizieren — das braucht einen **sparse/gather-Matmul-
Kernel**. In reinem Rust auf der GPU geht das nicht ohne Kernel; auf der **CPU**
ginge es (sparse Matmul in Rust), aber die CPU ist bei uns der Flaschenhals.
→ Bester pure-Rust-Nutzen: Sparsity auf dem **CPU-Offload-Pfad** einsetzen, damit
ausgelagerte Layer nur ihren aktiven Teil rechnen.

---

## 3. Kategorie C — Offloading-Systeme *(großes Modell passt, Teil läuft langsamer)*

Modell zwischen GPU und CPU/RAM (oder SSD) aufteilen. Verlustfrei; die Kunst ist,
den langsamen Pfad (PCIe, CPU) klein zu halten oder zu überlappen.

- **PowerInfer** (SOSP-nah, 2023/24) — GPU/CPU-Hybrid über Aktivierungs-Sparsity
  + Offline-Neuron-Placement; bis zu **11.69×** über llama.cpp auf Consumer-GPU.
  → <https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf>
- **PowerInfer-2** — dieselbe Idee auf Smartphones. → Repo:
  <https://github.com/Tiiny-AI/PowerInfer>
- **llama.cpp** — der Standard: ganze Layer auf GPU, Rest auf CPU. Durch **PCIe +
  schwache CPU** limitiert → genau unsere ~2–3 tok/s-Prognose.
- **DynamicInfer** (ICLR 2026) — „runtime-aware sparse", dynamisches Placement zur
  Laufzeit. → <https://openreview.net/pdf/e9b26b26d8a26d5844d0e52a27b30b35f86e875d.pdf>
- **PIPO** — pipelined Offloading für Consumer-Geräte (Transfer/Compute-Overlap).
  → <https://arxiv.org/pdf/2504.03664>
- **FlexInfer** — flexibles/effizientes Offloading für On-Device-Inferenz.
  → <https://arxiv.org/pdf/2503.03777>
- **NEO** — CPU-Offloading gegen die „GPU-Memory-Crisis" im Online-Serving.
  → <https://arxiv.org/pdf/2411.01142>
- **FlexGen** — Hochdurchsatz-Inferenz großer Modelle auf *einer* GPU (batch-,
  nicht latenz-orientiert).

**Verdict für uns:** ✅ **Der robuste Fit-Weg, in reinem Rust baubar** (ganze
Layer GPU/CPU, bestehende Kernels wiederverwenden). Ehrlicher Speed: ~2–3 tok/s
beim 15-GB-Modell, weil die **CPU der Flaschenhals** ist. PowerInfers 11× gelten
für ReLU-Modelle + eigene Kernels + bessere CPU/GPU-Balance — **nicht** auf unsere
Lage übertragbar. **Direkter pure-Rust-Hebel:** den **CPU-Matmul** in
`poezi-tensor` beschleunigen (SIMD/Cache-Blocking/Threads) — hebt *jedes*
ausgelagerte Layer.

---

## 4. Kategorie D — Spekulatives / paralleles Decoding *(weniger Token-Pässe, verlustfrei)*

Mehrere Token auf einmal erzeugen und in einem Pass prüfen. Bei temp 0 verlustfrei.
**Haben wir schon** (Prompt-Lookup, 1.67× auf Reuse-Workloads).

- **Prompt-Lookup / Lookahead** — kein zweites Modell; Entwürfe aus dem Kontext
  bzw. n-Gramm-Jacobi. Lookahead ~1.12×. **(unser Ansatz)**
- **Medusa** — leichte „Draft-Heads" direkt am Zielmodell (~1.46×).
- **EAGLE / EAGLE-2 / EAGLE-3** — aktueller SOTA. EAGLE-3 lässt die
  Feature-Prediction fallen, sagt Token direkt voraus, nutzt fusionierte
  Multi-Layer-Features; EAGLE gesamt ~1.98×.
  → EAGLE-3: <https://arxiv.org/pdf/2503.01840> · EAGLE:
  <https://arxiv.org/pdf/2401.15077> · EAGLE-2:
  <https://arxiv.org/pdf/2406.16858>
- Survey (EMNLP-Findings 2025): <https://aclanthology.org/2025.findings-emnlp.716.pdf>

**Verdict für uns:** ✅ **Erledigt mit der einfachsten (trainingsfreien)
Variante.** EAGLE-3 wäre stärker (~2×), braucht aber ein **trainiertes Draft-Head**
— das passt sogar zu unserer QLoRA-Pipeline und *nicht* zwingend zu einem
CUDA-Kernel (ist reines Modell-Zeug). **Realistischster nächster Decoding-Schritt,
wenn überhaupt.**

---

## 5. Kategorie E — KV-Cache-Kompression *(spart Speicher bei langen Kontexten)*

Betrifft nicht die Gewichte, sondern den mit der Kontextlänge wachsenden
KV-Cache — relevant für lange Agent-Sessions.

- **StreamingLLM** — nur erste k + letzte α Token behalten (Attention-Sinks).
- **H2O / SnapKV** — wichtige KV-Positionen pro Head auswählen/behalten.
- **PyramidKV** — pro Layer unterschiedlich stark kürzen (tiefere Layer weniger).
  → <https://arxiv.org/pdf/2406.02069>
- **ZigZagKV** — dynamisch nach Layer-Unsicherheit. → <https://arxiv.org/pdf/2412.09036>
- **KIVI** — KV **quantisieren** (nicht unser Pfad, da Qualität).
- **THINK** — query-abhängiges KV-Pruning, ~20 % über reine Eviction.

**Verdict für uns:** ⚠️ **Später relevant.** Nützlich, wenn OVERFLOW-Agenten über
den ~4096-Token-Cache hinaus laufen. Eviction/Selektion ist reiner Rust und
verlustarm (nicht verlustfrei). Für das 15-GB-Fit-Problem **nicht** der Hebel.

---

## 6. Kategorie F — Adaptive Rechentiefe & effiziente Architekturen *(weniger Rechnung, aber meist neues Modell/Training)*

- **Mixture-of-Depths** (Raposo 2024) — Top-k-Routing: nicht jedes Token
  durchläuft jeden Layer.
- **Mixture-of-Recursions** (2025) — geteilter Layer-Stack, pro Token dynamische
  Rekursionstiefe; vereint Parameter-Effizienz + adaptives „Nachdenken".
  → <https://arxiv.org/pdf/2507.10524>
- **AdaPonderLM** (2026) — gated „pondering" mit token-weiser Tiefe.
  → <https://arxiv.org/pdf/2603.01914>
- **Early-Exit** — frühzeitig aussteigen, wenn ein Token „sicher" ist.

**Verdict für uns:** ❌/⚠️ **Meist Architektur-/Trainingssache** — braucht ein
anders trainiertes Modell. Early-Exit-artiges ließe sich *perspektivisch* mit
unserer Fine-Tuning-Pipeline aufsetzen, ist aber ein großes, unsicheres Projekt.

---

## 7. Kategorie G — MoE statt dicht *(architektonisch sparse, verlustfrei)*

Wenn „groß & schnell" das Ziel ist, ist die sauberste Antwort oft **gar kein
dichtes Modell**, sondern MoE: 15 GB gesamt, aber nur ~3 GB pro Token aktiv.

- **MoE-Infinity** — Experten in RAM/SSD auslagern, heiße Experten GPU-resident,
  Tracing + Prefetch versteckt Transfers. → <https://github.com/EfficientMoE/MoE-Infinity/>
- **DALI** — workload-bewusstes MoE-Offloading für lokale PCs.
  → <https://arxiv.org/pdf/2602.03495>
- **HybriMoE** (DAC 2025) — CPU/GPU-Scheduling + Cache für MoE.
- **fMoE** — fein-granulares Experten-Offloading.

**Verdict für uns:** ⚠️ **Strategisch sehr interessant, aber anderes Modell.**
Ein MoE-Code-Modell (z. B. Qwen3-MoE-Klasse) mit Experten-Offloading könnte „groß
& flott auf 12 GB" besser lösen als dichtes Offloading — der aktive Satz passt in
den VRAM. Setzt voraus, dass wir von dichtem Qwen2.5 weggehen.

---

## 8. Kategorie H — Domänen-Spezialisierung *(der Teil, den ich fälschlich „neu" nannte)*

Wichtig fürs ehrliche Bild: Die Idee „unser Agent ist eng (Security/Code), also
ist sein aktiver Neuron-/Experten-Satz kleiner und stabiler" **gibt es bereits**.

- **SEAP** — training-free „Sparse Expert Activation Pruning": identifiziert und
  erhält **task-spezifische** Neuronen über Aktivierungsmuster.
  → <https://lacuna.tiptreesystems.com/work/seap-training-free-sparse-expert-activation-pruning-unlock-the-brainpower-of/>
- **DieT — „Diet Your LLM"** — profiliert Aktivierungs-Magnituden über Tasks mit
  **nur ~100 Samples/Task** und baut per Mehrheitsvotum eine globale Maske.
  → <https://arxiv.org/html/2603.23985v1>
- **„Half the Experts, All the Code"** — One-Shot-Domänen-Pruning von **MoE** fürs
  Coding. → <https://arxiv.org/pdf/2607.16721>
- **Domain-Specific Expert Pruning for MoE** → <https://proceedings.mlr.press/v317/yao26a.html>
- **TrimLLM** — progressives Layer-Dropping für domänenspezifische LLMs (ACL 2025).

**Verdict für uns:** Unsere „Domänen-Hot-Satz"-Idee ≈ **DieT/SEAP**. Also **nicht
neu** als Konzept. Was daran für uns offen bleibt, steht unten.

> **Eigenes Experiment (2026-10-04, `poezi domaintest`) — NEGATIVES ERGEBNIS.**
> Wir haben einen Hot-Satz aus Security/Code-Prompts gelernt und seine Abdeckung
> auf *anderen* Security/Code-Prompts vs. generischem Text gemessen (hot = 25 %):
> **Domäne 36.1 % vs. generisch 33.0 % — Gap nur +3.0 pp**, und nicht besser als
> die ~39 %-Generik-Baseline. Der Hot-Satz ist also **nicht domänenspezifisch**.
> → Statisches Domänen-Offloading ist bei diesem dichten SiLU-Modell **tot**,
> genauso wie generisches statisches Placement. Die eine potenziell originelle
> Idee ist damit empirisch widerlegt — ehrlich festgehalten.

---

## 9. Ehrliche Synthese — was ist bekannt, was wäre *wirklich* neu für uns?

**Bekannt (= Nachbauen, kein Forschungsbeitrag):** spekulatives Decoding,
Aktivierungs-Sparsity auf SwiGLU (TEAL/CATS), Prädiktoren (SparseInfer), statisches
& dynamisches Offloading (PowerInfer, llama.cpp, DynamicInfer), verlustfreie
Kompression (DFloat11), KV-Kompression, MoE-Offloading, Domänen-Pruning (SEAP/DieT).
Praktisch jeder einzelne „Hebel", den wir diskutiert haben, ist publiziert.

**Die echte, unterbeleuchtete Lücke liegt nicht in einer neuen *Technik*, sondern
in unserem *Regime*:**
1. **Schwache GPU + sehr langsame CPU + KEINE eigenen Kernels.** Fast alle
   Speedups der Literatur setzen Custom-Kernels (sparse Gather, Huffman-Decode,
   fused MoE) und/oder High-End-GPUs voraus. Ob diese Methoden unter *reiner
   Orchestrierung* (kein Kernel) auf einer 3060 **netto** noch gewinnen, ist
   kaum untersucht und messtechnisch offen.
2. **Ein einzelner, eng spezialisierter, lokal fine-getunter Agent** (nicht ein
   Generalist auf Benchmarks). *Unser Test (§8) zeigt: die Domäne allein
   erzeugt KEINE nutzbare Struktur (+3 pp).* Die Struktur ist also nicht da —
   **man müsste sie hineintrainieren.** Die eine verbliebene originelle Richtung:
   **Sparsity aktiv ins Fine-Tuning hineintrainieren** (Richtung CATS/
   ReLUfication, aber als Teil *unserer* QLoRA-Schleife), sodass der Agent
   *lernt*, scharf & stabil sparse zu feuern — das Gegenteil von „vorhandene
   Sparsity ausnutzen". Ambitioniert und unsicher, aber genuin unser Beitrag.
3. **Ein ehrlich gemessener, kombinierter Stack** für genau dieses Gerät:
   verlustfreie Kompression (Fit) × Aktivierungs-Sparsity (Rechnung) ×
   spekulatives Decoding (Pässe), end-to-end auf einer 3060 vermessen. So eine
   saubere Kombinations-Messung im Low-End-Regime ist eher „Systems"-Beitrag als
   neue Theorie — aber real und für andere nützlich.

**Nüchterne Einordnung:** Einen echten *algorithmischen* Durchbruch sollten wir
nicht versprechen — daran arbeiten große Gruppen. Realistisch ist ein **origineller
Systems-/Empirie-Beitrag** im Low-End-/Single-Agent-Regime, plus ein
möglicherweise neuer Dreh beim **sparsity-aware Fine-Tuning** unseres eigenen
Agenten.

---

## 10. Empfohlene Forschungsagenda (konkret, messbar, mit ehrlichem Risiko)

Reihenfolge nach „Nutzen fürs Ziel ÷ Risiko":

1. **CPU-Matmul in `poezi-tensor` beschleunigen** *(reiner Rust, risikoarm, hoher
   Hebel)*. Der CPU-Pfad limitiert jedes Offloading. SIMD (AVX2), Cache-Blocking,
   besseres Threading. Messbar via `poezi gen` CPU-tok/s. **Fundament für alles
   Offloading.**
2. **Whole-Layer-Offloading** *(reiner Rust, Fit-Ziel)*. Erreicht „15 GB in 12 GB,
   volle Qualität" jetzt, ~2–3 tok/s (mit Schritt 1 schneller).
3. **Sparsity auf dem CPU-Offload-Pfad** *(TEAL/CATS-Idee, reiner Rust auf CPU)*.
   Ausgelagerte FFN-Layer nur ihren aktiven ~40–50 %-Teil rechnen lassen —
   verlustfrei laut unserem Masktest, und es trifft genau den langsamen Pfad.
4. **Sparsity-aware Fine-Tuning** *(nutzt unsere QLoRA-Pipeline — möglicher
   Eigenbeitrag)*. Den Agenten so fine-tunen, dass seine FFN-Aktivierungen
   *schärfer* und *stabiler* werden (Richtung CATS/SEAP, aber in-the-loop für
   unsere Domäne). Vorher mit `poezi skew` auf **echten Security/Code-Prompts**
   messen, ob die Domäne den Hot-Satz schon konzentriert.
5. **DFloat11-artige verlustfreie Kompression** *(nur falls du die „keine
   Kernels"-Regel hierfür lockern willst)* — der sauberste Fit-Weg für ein
   *unquantisiertes* großes Modell.
6. **MoE-Pfad evaluieren** *(strategisch)* — ein MoE-Code-Modell + Experten-
   Offloading könnte „groß & schnell auf 12 GB" fundamental besser lösen.

---

## 11. Quellen

**Verlustfreie Kompression:** DFloat11 <https://arxiv.org/pdf/2504.11651>,
<https://github.com/LeanModels/DFloat11> · Shannon-Bound
<https://arxiv.org/pdf/2606.15789>
**Aktivierungs-Sparsity:** TEAL <https://arxiv.org/pdf/2408.14690>,
<https://iclr.cc/virtual/2025/poster/28994> · SparseInfer
<https://arxiv.org/pdf/2411.12692> · Polar Sparsity
<https://arxiv.org/pdf/2505.14884> · First Activations Matter
<https://arxiv.org/pdf/2408.11393>
**Offloading:** PowerInfer
<https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf>,
<https://github.com/Tiiny-AI/PowerInfer> · DynamicInfer
<https://openreview.net/pdf/e9b26b26d8a26d5844d0e52a27b30b35f86e875d.pdf> · PIPO
<https://arxiv.org/pdf/2504.03664> · FlexInfer
<https://arxiv.org/pdf/2503.03777> · NEO <https://arxiv.org/pdf/2411.01142>
**Spekulatives Decoding:** EAGLE-3 <https://arxiv.org/pdf/2503.01840> · EAGLE
<https://arxiv.org/pdf/2401.15077> · EAGLE-2 <https://arxiv.org/pdf/2406.16858> ·
Survey <https://aclanthology.org/2025.findings-emnlp.716.pdf>
**KV-Cache:** PyramidKV <https://arxiv.org/pdf/2406.02069> · ZigZagKV
<https://arxiv.org/pdf/2412.09036>
**Adaptive Tiefe:** Mixture-of-Recursions <https://arxiv.org/pdf/2507.10524> ·
AdaPonderLM <https://arxiv.org/pdf/2603.01914>
**MoE-Offloading:** MoE-Infinity <https://github.com/EfficientMoE/MoE-Infinity/> ·
DALI <https://arxiv.org/pdf/2602.03495>
**Domänen-Spezialisierung:** SEAP
<https://lacuna.tiptreesystems.com/work/seap-training-free-sparse-expert-activation-pruning-unlock-the-brainpower-of/>
· Diet Your LLM <https://arxiv.org/html/2603.23985v1> · Half the Experts, All the
Code <https://arxiv.org/pdf/2607.16721> · Domain-Specific Expert Pruning
<https://proceedings.mlr.press/v317/yao26a.html>

---

*Nächster Schritt laut Agenda: Punkt 1 (CPU-Matmul) oder zuerst `poezi skew` auf
echten Security/Code-Prompts, um die Domänen-Hypothese zu prüfen. Entscheidung
liegt bei S1BERIA.*
