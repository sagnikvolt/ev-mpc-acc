# S1b model-mismatch sensitivity (Opus, written 2026-11-09 before running). No pass/fail; reported as robustness.
Concern: MPC-map-L's lead model (a_dec 2.0, a_acc 1.0, vb 4.2) equals the generator exactly. Test: real lead follows different rules, MPC keeps assuming 2.0/1.0/4.2.
Variants: M1 lead a_dec 1.2, vb 3.0 (cautious driver, slows early and lower); M2 lead a_dec 3.0, vb 5.5 (late, faster crossing). First 14 S1b test windows, same seeds. Arms ACC-map, MPC-map-L.
Report median dWh/km, unsafe, crossing excess. Claim survives if MPC-map-L stays below ACC-map with 0 extra unsafe windows in both.
