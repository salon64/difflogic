# Reading List — Memory in Sequential Systems (not just papers)

_Created 2026-07-01. Curated, prioritized learning list for Paper #2
([11_paper2_workmap.md](11_paper2_workmap.md)): differentiable logic gates + hardware memory
primitives (D / SR / JK / T flip-flops, latches) → a trained clocked circuit that is an FSM.
Emphasis on **textbooks, courses, lectures, blogs, interactive tools** — not only papers.
Every link was confirmed live as of 2026-06-30; access caveats flagged at the end._

**The throughline:** *memory in a sequential system is a bit held stable by feedback until
an enable/clock permits change.* P2 needs five lenses on that one idea — the hardware
primitive (A), how it maps to real fabric (B), its soft/gradient analog (C), the
fixed-point/bistability theory that makes feedback store a bit (D), explicit addressable
memory as background (E), and how to read the learned FSM back out (F).

---

## ⭐ If you only read 5 things (best-first, spanning all lenses)

1. **Harris & Harris, _Digital Design and Computer Architecture_** (+ free video lectures) —
   builds *every* primitive you're relaxing (bistable → SR → D-latch → D-FF → FSM) up from
   gates, the exact gate-composition mindset of difflogic. → the hardware primitive.
   <https://pages.hmc.edu/harris/ddca/ddcarv/ddcarv_videos.html>
2. **Olah, "Understanding LSTM Networks"** — the canonical gating intuition: a cell state
   held unchanged while a multiplicative gate decides whether to overwrite it — the soft,
   differentiable version of a clocked latch's enable. → the ML analog.
   <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
3. **Kolter, Duvenaud & Johnson, "Deep Implicit Layers" (NeurIPS 2020 tutorial)** — the
   precise math + code for differentiating through a feedback loop that settles to a fixed
   point, which is exactly a combinational-feedback latch. → the theory that makes feedback
   memory trainable (our Tier-3 section). <http://implicit-layers-tutorial.org/>
4. **Weiss, Goldberg & Yahav, "Extracting Automata from RNNs Using Queries and
   Counterexamples" (ICML 2018)** — the exact, runnable algorithm to recover the DFA a
   recurrent model learned; for our circuit this is literally how you read out and verify
   the learned FSM. → reading the FSM back out.
   <https://arxiv.org/abs/1711.09576>
5. **Nand2Tetris, _Build a Modern Computer from First Principles_** — closest in spirit to
   the project: compose one gate into DFF → register → RAM → a clocked CPU, treating the DFF
   as the single primitive that "introduces time/state." → the build mindset.
   <https://www.nand2tetris.org/>

---

## A. Digital sequential logic fundamentals
_Flip-flops (D/SR/JK/T), latch vs flip-flop, Mealy/Moore FSMs, state encoding, metastability,
setup/hold, clocking._

1. **Digital Design and Computer Architecture** — Harris & Harris — *textbook + free video
   course.* Ch.3 video sequence: Bistable → SR latch → D latch → D-FF → FF variations →
   synchronous sequential logic → Mealy/Moore → timing → clock skew → metastability →
   synchronizers; Ch.4 implements FSMs in SystemVerilog. **Why:** the single best 1:1 map to
   the project — builds each differentiable primitive from gates, then composes flip-flops
   into the exact Mealy/Moore FSM a sequential LGN *is*.
   Hub <https://pages.hmc.edu/harris/ddca/ddcarv.html> · Videos
   <https://pages.hmc.edu/harris/ddca/ddcarv/ddcarv_videos.html> · edX audit
   <https://www.edx.org/learn/design/harvey-mudd-college-digital-design>
2. **Digital Design and Computer Architecture @ ETH Zürich** — Onur Mutlu — *free full
   university course* (uses Harris & Harris). Slides, labs, homeworks; deeper timing /
   metastability. **Why:** courseware to *practice* deriving next-state logic and excitation
   tables — the functions each learnable memory cell must approximate. (Also the ETH group
   flagged as the sequential-LGN race-risk.) Playlist
   <https://www.youtube.com/playlist?list=PL5Q2soXY2Zi9Eo29LMgKVcaydS7V1zZW3> · Hub
   <https://people.inf.ethz.ch/omutlu/lecture-videos.html>
3. **Nand2Tetris** — Nisan & Schocken — *project-based course.* Build a computer from one
   Nand: gates → **DFF → registers → RAM** → FSM-driven CPU, in a free HDL simulator.
   **Why:** treats the DFF as the one irreducible primitive that introduces state — exactly
   the insight behind adding a differentiable flip-flop to a feed-forward LGN.
   <https://www.nand2tetris.org/> · Coursera audit
   <https://www.coursera.org/learn/build-a-computer>
4. **MIT 6.004 Computation Structures** — *free course (OCW).* Transistors → gates →
   sequential logic → FSMs → CPU; strong on the "dynamic discipline" (setup/hold, why
   feedback + clock gives stable state). **Why:** the timing theory justifying modeling a
   recurrent LGN as clean discrete steps rather than messy analog feedback.
   <https://ocw.mit.edu/courses/6-004-computation-structures-spring-2017/>
5. **Ben Eater — breadboard 8-bit computer** — *hands-on video series.* Physically builds SR
   latch, gated D latch, edge-triggered registers, and a clock module from 7400-series chips.
   **Why:** unmatched physical intuition for what a register *does* on a clock edge — the
   concrete referent for the abstraction you're relaxing. <https://eater.net/8bit>
6. **All About Circuits — _Digital_** — *free interactive textbook.* Ch.10 (latches/
   flip-flops) + Ch.11 (FSMs, counters) with truth/excitation tables. **Why:** fast free
   reference for the **excitation tables** — the precise next-state function each
   differentiable SR/JK/T cell must emulate.
   <https://www.allaboutcircuits.com/textbook/digital/chpt-11/finite-state-machines/>
7. **Neso Academy — Digital Electronics** — *free YouTube playlist.* Granular per-topic
   videos (NAND/NOR SR latch, triggering, D/JK/T, clocked-sequential analysis). **Why:**
   best for targeted gap-filling on one primitive/variant at a time.
   <https://www.youtube.com/playlist?list=PLBlnK6fEyqRjMH3mWf6kwqiTbT798eAOm>
8. **Mano & Ciletti, _Digital Design_** and **Wakerly, _Digital Design: Principles and
   Practices_** — *textbooks (free Internet Archive borrow).* Mano = gold standard on **state
   encoding (binary/one-hot/Gray), state minimization, Mealy/Moore**; Wakerly = most
   practical on **metastability / setup-hold / synchronizers** (matters for FPGA deployment).
   Mano <https://archive.org/details/digital-design-5th-edition-m-morris-mano-and-michael-d-ciletti>
   · Wakerly <http://wakerly.org/DDPP/index4e.html>

**Single-concept explainers:** latch vs flip-flop (level vs edge) —
<https://www.electronicsforu.com/technology-trends/latch-not-bad-latch-vs-flip-flop> (this
level-vs-edge choice is *the* key design decision: a transparent/latch-like soft cell
creates combinational feedback that can race in the relaxed regime; an edge-triggered cell
gives a clean discrete update) · metastability + setup/hold —
<https://nandland.com/lesson-13-metastability/> · Mealy vs Moore —
<https://www.geeksforgeeks.org/difference-between-mealy-machine-and-moore-machine/> (your
model literally *is* one of these — a direct choice for where the output head reads from).

**Interactive simulators:** Falstad CircuitJS (animated SR / master-slave / edge-triggered)
<https://www.falstad.com/circuit/> · CircuitVerse <https://circuitverse.org/> · Logisim
Evolution (offline, RAM/ROM/subcircuits) <https://github.com/logisim-evolution/logisim-evolution>.

---

## B. How memory maps to hardware / HDL
_Register/FF inference, `always @(posedge clk)`, what synthesizes to a FF vs a latch (and why
inferred latches are a bug), FPGA LUT+FF fabric._

1. **Nandland — FPGA-101 + blog** — *tutorial/blog.* "Flip-flop = register," the rising-edge
   clock, LUT+FF as the two fundamental FPGA components, blocking-vs-nonblocking, latch
   avoidance. **Why:** the best conceptual on-ramp — the mental model the RTL emitter needs
   ("a memory primitive is a D-FF, emitted as `always @(posedge clk)` with a nonblocking
   assignment"). <https://nandland.com/fpga-101/> · latch avoidance
   <https://nandland.com/how-to-avoid-creating-a-latch/> · blocking vs nonblocking
   <https://nandland.com/blocking-vs-nonblocking-in-verilog/>
2. **HDLBits — interactive Verilog problem set** — *interactive tool, KEY.* ~180 auto-graded
   exercises with dedicated D-FF / register / latch / FSM sections. **Why:** fastest way to
   internalize register inference and *see* latch inference happen — drills the exact
   patterns the trained circuit will be emitted as.
   <https://hdlbits.01xz.net/wiki/Problem_sets>
3. **ZipCPU (Gisselquist) — tutorial + blog** — *tutorial/blog.* Correctness-first Verilog
   via open tools (Verilator + SymbiYosys formal); "Rules for newbies" = the canonical
   synchronous-design discipline (one clock, register only on `posedge clk`, never gate a
   clock). **Why:** load-bearing for real-FPGA targeting — the rules a clocked LGN must obey,
   in the exact open verify flow you'd run on generated RTL.
   <https://zipcpu.com/tutorial/> · <https://zipcpu.com/blog/2017/08/21/rules-for-newbies.html>
4. **AMD/Xilinx UG901 — Vivado Synthesis** — *vendor docs.* Ground-truth HDL→hardware
   inference rules; "Flip-Flops, Registers, and Latches" gives rising-edge templates and the
   incomplete-`if`/`case` → inferred-latch warning. **Why:** the spec the emitter must
   conform to so trained D-FFs synthesize as registers, not accidental latches.
   <https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Flip-Flops-Registers-and-Latches>
5. **Intel/Altera — Recommended HDL Coding Styles** — *vendor docs.* Cross-vendor
   confirmation; register power-up values and secondary control (async clear, clock-enable).
   **Why:** the clear/enable section maps directly to how SR/JK/T reset/enable should be
   coded.
   <https://www.intel.com/content/www/us/en/docs/programmable/683082/22-1/recommended-hdl-coding-styles.html>
6. **Cummings, "Nonblocking Assignments in Verilog Synthesis — Coding Styles That Kill!"** —
   *reference paper (MIT mirror).* The definitive rule: nonblocking (`<=`) for sequential,
   blocking (`=`) for combinational, never mix in one block. **Why:** governs how the emitter
   writes D-FFs — get it wrong and the emitted circuit simulates differently than it
   synthesizes. <https://csg.csail.mit.edu/6.375/6_375_2009_www/papers/cummings-nonblocking-snug99.pdf>
7. **Inferred latches / FPGA fabric** — *blogs.* Doulos on transparent latches (incomplete
   `if`/`case` → inferred latch → wrecked timing)
   <https://www.doulos.com/knowhow/fpga/why-should-i-care-about-transparent-latches/> ·
   Digilent on the **LUT+FF slice** (where a trained LGN lands: gate logic fills LUTs, each
   memory primitive consumes a slice flip-flop)
   <https://digilent.com/blog/fpga-configurable-logic-block/>.
8. **ASIC-World Verilog examples** — *reference* — copy-paste D/T/SR/JK templates to diff the
   emitter against: <https://www.asic-world.com/examples/verilog/flip_flop.html>

**Validate emitted RTL:** EDA Playground (browser, Icarus + Yosys)
<https://edaplayground.com/> · Verilator (fast sim + latch-prone lint)
<https://verilator.org/guide/latest/> · *paid:* Pong P. Chu, _FPGA Prototyping by Verilog
Examples_.

---

## C. Memory in recurrent neural networks (ML side)
_Constant error carousel, vanishing gradients, gating, copy/adding long-range benchmarks —
explainer-level first._

1. **"Understanding LSTM Networks"** — Christopher Olah — *blog.* Cell state as a "conveyor
   belt" + forget/input/output gates, clearly diagrammed. **Why:** the core gating intuition
   — a stored value held until a multiplicative gate permits change is the soft analog of a
   clocked latch's enable. Start here.
   <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
2. **"The Unreasonable Effectiveness of RNNs"** — Andrej Karpathy — *blog.* Char-RNN +
   neuron-level state visualizations. **Why:** builds the "an RNN is a stateful circuit
   unrolled over clock ticks" model; the neuron that latches on/off across a quote is a vivid
   picture of one learned memory bit persisting.
   <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>
3. **Deep Learning, Ch.10 (Sequence Modeling)** — Goodfellow, Bengio & Courville — *free
   textbook.* BPTT, the formal vanishing/exploding analysis (recurrent-Jacobian
   eigenvalues), LSTM/GRU/gated units. **Why:** the math for *why* ungated recurrence
   destroys long-range gradients and why a gated linear self-loop fixes it — the theoretical
   justification for a memory primitive.
   <https://www.deeplearningbook.org/contents/rnn.html>
4. **"Illustrated Guide to LSTM's and GRU's"** — Michael Phi — *blog + video.* Animated,
   minimal-math flow through each gate. **Why:** clearest first pass on *which gate does what
   to stored state* — gates opening/closing map almost 1:1 onto a register's write-enable.
   <https://www.youtube.com/watch?v=8HyCNIVRbSU>
5. **"Written Memories: Understanding, Deriving and Extending the LSTM"** — R2RT — *blog.*
   Derives LSTM/GRU from first principles, analyzes vanishing gradients, connects to
   highway/residual nets and NTMs. **Why:** best resource for *designing a new memory
   primitive* — shows the property you must engineer (an uninterrupted gradient path).
   <https://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html>
6. **CS224n L7, "Vanishing Gradients and Fancy RNNs"** — Stanford — *course (video + notes).*
   **Why:** cleanest lecture treatment of the exact failure mode a flip-flop sidesteps by
   holding state losslessly across cycles. <https://www.youtube.com/watch?v=QEw0qEa0E50>
7. **"Long Short-Term Memory" (1997)** — Hochreiter & Schmidhuber — *paper.* The **Constant
   Error Carousel**: a self-connected weight-1.0 unit that holds state without decay, fenced
   by multiplicative gates. **Why:** the conceptual seed of the project — the CEC is the
   continuous ancestor of a clocked latch that holds a bit until its enable fires. Read the
   CEC section. <https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf>
8. **"Learning Long-Term Dependencies… Is Difficult" (1994)** — Bengio, Simard & Frasconi —
   *paper.* The origin vanishing-gradient result: a fundamental trade-off between robustly
   storing information and learning it by gradient descent. **Why:** grounds the argument
   that a discrete latch sidesteps a problem provably intrinsic to gradient-trained
   continuous recurrence.
   <https://www.semanticscholar.org/paper/Learning-long-term-dependencies-with-gradient-is-Bengio-Simard/d0be39ee052d246ae99c082a565aba25b811be2d>
9. **The copy/adding benchmarks** — Le, Jaitly & Hinton, "IRNN" (2015) — *paper.* Clean
   definitions of the **adding problem** and **copy/memory task**. **Why:** the benchmark
   source — copy is precisely the test that proves a sequential logic net can store a bit
   across time. <https://arxiv.org/abs/1504.00941>
   - *Diagnostic bonus:* Distill, "Visualizing Memorization in RNNs" — a "connectivity" viz
     to *measure* whether stored info truly persists. <https://distill.pub/2019/memorization-in-rnns/>

---

## D. Fixed-point / equilibrium / associative memory
_A latch = a bistable fixed point of cross-coupled feedback. The theory of how feedback
stores a bit._

1. **"Deep Implicit Layers — Neural ODEs, Deep Equilibrium Models, and Beyond"** — Kolter,
   Duvenaud & Johnson (NeurIPS 2020) — *interactive tutorial + video + Colab (JAX & PyTorch).*
   Ch.2 (implicit differentiation) + Ch.4 (DEQ) are core. **Why:** the precise machinery for
   differentiating through a feedback loop that settles to a fixed point — exactly a
   combinational-feedback latch; the single most directly applicable Tier-3 tool.
   <http://implicit-layers-tutorial.org/> · video <https://www.youtube.com/watch?v=MX1RJELWONc>
2. **"Hopfield Networks is All You Need" — companion blog** — Ramsauer et al. (ML-JKU) —
   *blog + paper + code.* Modern continuous-state Hopfield energy + one-step update, from
   classical discrete Hopfield nets. **Why:** clearest modern tutorial on **energy → update
   rule → attractor retrieval**; its discrete-Hopfield foundation is the canonical model of a
   bit stored as a bistable attractor — the conceptual core of a latch (and the §D.3 energy
   framing in the work map). <https://ml-jku.github.io/hopfield-layers/>
   *(note: `hopfield.dev` is an unrelated company, NOT this explainer.)*
3. **Scholarpedia: Hopfield network** — John J. Hopfield — *encyclopedia.* Lyapunov/energy
   function, stable states, content-addressable memory. **Why:** shortest rigorous route to
   "a stored bit = a stable minimum of an energy function" — underwrites framing an SR-latch's
   two states as two attractors of cross-coupled gates.
   <http://www.scholarpedia.org/article/Hopfield_network>
4. **Information Theory, Inference, and Learning Algorithms, Ch.42** — David MacKay — *free
   textbook.* Hopfield nets as associative memory, energy, storage capacity, Ising/Boltzmann.
   **Why:** rigorous-yet-readable derivation of attractor dynamics and *how many* distinct
   bits a feedback structure can hold. <https://www.inference.org.uk/itprnn/book.pdf>
5. **Deep Equilibrium Models (DEQ)** — Bai, Kolter & Koltun (NeurIPS 2019) — *paper + code.*
   Infinite-depth weight-tied net as a single root-finding/fixed-point solve, with
   constant-memory implicit-diff backprop. **Why:** the canonical "the whole layer *is* a
   fixed point of a feedback map" model — the template for treating a latch's settled state
   as the equilibrium you train through (and the thing that *breaks* on bistability — work
   map §D). <https://arxiv.org/abs/1909.01377> · code <https://github.com/locuslab/deq>
6. **"A Tutorial on Energy-Based Learning"** — LeCun, Chopra & Hadsell — *tutorial paper +
   lectures.* Energy per configuration; inference = clamp + minimize. **Why:** vocabulary for
   a latch's stable states as low-energy basins and switching as crossing a barrier.
   <http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf>
7. **Hopfield (1982), "Neural networks and physical systems…"** — *foundational paper.* The
   original content-addressable, error-correcting attractor memory. **Why:** the origin of
   memory-as-attractor (read once; tutorials above teach it faster).
   <https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield82.pdf>
8. **Neuromatch Academy W2D4 "Dynamic Networks" + Scholarpedia "Attractor network"** —
   *hands-on Colab + encyclopedia.* Phase-plane analysis of a 2-population feedback system:
   fixed points, **bistability**, oscillations; point/line/ring attractors. **Why:** the most
   direct dynamical-systems intuition for "two stable fixed points of a feedback loop store
   one bit."
   <https://compneuro.neuromatch.io/tutorials/W2D4_DynamicNetworks/student/W2D4_Tutorial2.html>
   · <http://www.scholarpedia.org/article/Attractor_network>
   - *Adjacent:* Neural ODEs (Chen et al. 2018), same implicit-layer machinery, continuous-time
     framing if you model latch settling as continuous relaxation. <https://arxiv.org/abs/1806.07366>

---

## E. Memory-augmented neural networks (background)
_Explicit, addressable memory bolted onto a neural controller. Shared problem with ours: make
a discrete memory op (address/read/write/latch) trainable by relaxing it, then harden it._

1. **"Attention and Augmented Recurrent Neural Networks"** — Olah & Carter (Distill) —
   *interactive blog.* Animated NTM content- vs location-based addressing, soft read/write
   heads, attention as the unifying trick. **Why:** shows *visually* how "pick location k"
   relaxes into a softmax over all locations so gradients flow — the exact relaxation to make
   a discrete flip-flop address/select differentiable. Start here.
   <https://distill.pub/2016/augmented-rnns/>
2. **"Explanation of Neural Turing Machines"** — Rylan Schaeffer — *technical explainer.*
   Reading/writing, content/location addressing, interpolation & sharpening, the five NTM
   tasks vs LSTM. **Why:** the mechanical detail level to design a differentiable
   latch/register and reason about its gradients.
   <https://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html>
3. **"Neural Turing Machines" — the morning paper** — Adrian Colyer — *prose explainer.*
   **Why:** good second read for the controller↔memory split — the template a Sequential LGN
   mirrors (logic-gate controller driving a bit-memory).
   <https://blog.acolyer.org/2016/03/09/neural-turing-machines/>
4. **"Neural Turing Machines"** — Graves, Wayne & Danihelka (2014) — *paper.* Neural
   controller + external memory via differentiable attentional heads. **Why:** the canonical
   precedent for "neural net + RAM by backprop"; the read/write head is the conceptual
   ancestor of a differentiable flip-flop. <https://arxiv.org/abs/1410.5401>
5. **Differentiable Neural Computer — DeepMind blog + Nature paper** — Graves et al. (2016) —
   *blog + paper + code.* Adds dynamic memory allocation, usage tracking, temporal link
   tracking (write *order*). **Why:** the temporal-link/allocation mechanisms are directly
   relevant to sequential state and a register-file-style memory.
   <https://deepmind.google/blog/differentiable-neural-computers/> · free PDF
   <https://web.stanford.edu/class/psych209/Readings/GravesWayne16DNC.pdf> · code
   <https://github.com/google-deepmind/dnc>
6. **RNN Symposium 2016 — Alex Graves on the DNC** — *talk/video.* **Why:** fastest way to
   absorb the design rationale you'll re-derive for logic-gate memory primitives.
   <https://www.youtube.com/watch?v=steioHoiEms>
7. **End-To-End Memory Networks** — Sukhbaatar et al. (2015) — *paper.* Recurrent soft-
   attention read over memory slots with multiple hops. **Why:** the cleanest minimal demo of
   making memory *access itself* differentiable (softmax over slots).
   <https://arxiv.org/abs/1503.08895> · lineage: Memory Networks (Weston et al. 2014)
   <https://arxiv.org/abs/1410.3916>

---

## F. Sequential circuits as learned automata
_The theoretical heart: RNNs-as-FSMs, extracting a DFA from a trained model, Tomita
benchmarks, computational power. The model literally learns an FSM-as-circuit._

1. **"Extracting Automata from RNNs Using Queries and Counterexamples"** — Weiss, Goldberg &
   Yahav (ICML 2018) — *paper + talk + slides + code.* Angluin's L\* (queries +
   counterexamples) with the trained RNN as oracle → an *exact* DFA of its state dynamics.
   **Why:** the single most directly applicable resource — the practical algorithm to read
   and verify the FSM a sequential LGN learned; the `lstar_extraction` repo is a ready
   template (drives the M4 FSM read-out in the work map).
   <https://arxiv.org/abs/1711.09576> · talk <https://youtu.be/ym64TaaHnT8?t=2766> · code
   <https://github.com/tech-srl/lstar_extraction>
2. **"Formal Abstractions of Neural Sequence Models" (ICGI 2021 tutorial)** — Gail Weiss —
   *tutorial video.* "What kind of machine is this network?" tying recurrent nets/transformers
   to automata. **Why:** the best accessible on-ramp to the neural-sequence-model-as-automaton
   mindset — watch first to frame everything else.
   <https://www.youtube.com/watch?v=J89zTx5vs7A>
3. **The 7 Tomita grammars** — *benchmark (code + DFA diagrams).* Binary-alphabet regular
   languages, DFAs of 3–6 states. **Why:** the de-facto suite for "did the model learn the
   right FSM?" — small automata a sequential LGN should learn, with ground-truth DFAs (the M4
   task). <https://github.com/LC-John/RNN2DFA>
4. **"Sequential Neural Networks as Automata"** — Merrill (ACL 2019 workshop) — *paper.*
   Real-time bounded-precision acceptance + a "network memory" measure; LSTMs behave like
   **counter machines**. **Why:** essentially the project's thesis — the formal vocabulary
   (state, memory, real-time acceptance) for arguing what automaton class a
   flip-flop-augmented network realizes. <https://aclanthology.org/W19-3901/>
5. **"A Formal Hierarchy of RNN Architectures"** — Merrill et al. (ACL 2020) — *paper + blog
   + talk.* Ranks recurrent cells by space (memory) and rational recurrence (weighted-FSM-
   ness); LSTM is *not* rational. **Why:** tells you which automaton class each cell expresses
   — directly relevant to what a D-FF (≈1 bit of finite state) buys you.
   <https://arxiv.org/abs/2004.08500> · blog
   <https://lambdaviking.com/blog/2020/formal-hierarchy-of-rnn-architectures/>
6. **"On the Practical Computational Power of Finite Precision RNNs"** — Weiss, Goldberg &
   Yahav (ACL 2018) — *paper + code.* Under finite precision + linear time, LSTMs can *count*
   and are strictly stronger than GRUs/squashing-RNNs (which collapse to finite-state).
   **Why:** the finite-state-vs-counter distinction in the exact regime a discrete logic
   circuit lives in — clarifies what pure-FSM sequential logic nets can/can't recognize.
   <https://aclanthology.org/P18-2117/>
7. **Giles et al. (1992) + Omlin–Giles (1996)** — *foundational papers.* Origin of RNN→DFA
   extraction; the 1996 paper proves you can *encode* a stable DFA directly into a recurrent
   net's weights. **Why:** both directions you care about — extraction (read the FSM out) and
   **encoding a DFA in** (the analogue of hand-building flip-flops into a logic-gate net,
   with provable stability).
   <https://clgiles.ist.psu.edu/pubs/NC1992-recurrent-NN.pdf>
8. **"Rule Extraction from RNNs: A Taxonomy and Review"** — Jacobsson (2005) — *survey.* The
   canonical map of pre-L\* extraction methods (quantization, clustering, partitioning) and
   their failure modes. **Why:** one-stop view of the design space and pitfalls before
   choosing an extraction strategy.
   <https://www.diva-portal.org/smash/get/diva2:2402/FULLTEXT01.pdf>
9. **"Thinking Like Transformers" / RASP** — Weiss, Goldberg & Yahav (ICML 2021) — *paper +
   interactive blog (adjacent).* A language whose primitives map onto attention/FFN — a net
   made legible as a program. **Why:** the gold standard of "programs-as-architecture"
   thinking; Sasha Rush's annotated RASPy is a model for presenting a learned logic circuit as
   a readable program. <https://srush.github.io/raspy/>
   - *Counterweight:* Hahn, "Theoretical Limitations of Self-Attention" (TACL 2020) —
     attention can't recognize PARITY/Dyck without growing depth/heads; sharpens which formal
     languages a memory-equipped sequential model handles that attention can't.
     <https://aclanthology.org/2020.tacl-1.11/>
   - *Closest prior art (train-soft / discretize-hard FSMs):* "Neural Networks as Universal
     Finite-State Machines: A Constructive DFA Theory" (2025) <https://arxiv.org/abs/2505.11694>
     · "GraphFSA" (2024) <https://arxiv.org/abs/2408.11042> — useful for situating the
     contribution and the sigmoid-during-training / round-at-inference regime that is exactly
     differentiable-logic-with-memory.

---

## Link caveats worth knowing
- **Dead/moved:** the historic Cummings nonblocking PDF on `sunburst-design.com` 404s → use
  the MIT mirror (B6). `hopfield.dev` is an unrelated company, **not** the ML-JKU Hopfield
  explainer (D2).
- **Loads in a browser but 403s automated fetchers** (not dead): safari.ethz.ch (Mutlu course
  page), Nandland latch page, Doulos, R2RT, inference.org.uk (MacKay), scholarpedia (http-only).
- **Flaky canonical PDFs → use the mirrors given:** LSTM-1997 (jku.at refused → CMU mirror,
  C7) and Bengio-1994 (umontreal TLS expired → Semantic Scholar, C8).
- **Paywalled with free alternates provided:** DNC *Nature* (free Stanford PDF, E5);
  Mano/Wakerly/Pong-Chu textbooks (Internet Archive borrow); Omlin–Giles JACM and Jacobsson
  MIT Press (free PSU/DiVA PDFs).
