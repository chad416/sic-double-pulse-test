# onsemi SPICE models (not included)

The onsemi EliteSiC M3S LTspice libraries are vendor-encrypted and carry the notice
"Permission is granted to use this file for simulations but not to reverse engineer
its contents". That grants use, not redistribution, so they are not committed here.

Download them yourself and place the ZIPs in this folder:

| File | Product page (Technical Documentation -> Simulation Models) |
|---|---|
| `onsemi_sicmosfet_1200m3s_ltspice.zip` | https://www.onsemi.com/products/discrete-power-modules/silicon-carbide-sic/silicon-carbide-sic-mosfets/nth4l022n120m3s |
| `onsemi_sicmosfet_650m3s_ltspice.zip` | https://www.onsemi.com/products/discrete-power-modules/silicon-carbide-sic/silicon-carbide-sic-mosfets/nth4l032n065m3s |

Then run `python -m dpt.fetch_models`. It verifies the SHA256 against the versions used
for the committed results (1200 V library v2.3, Sep 2025; 650 V library v2.6, Mar 2026)
and unpacks them. A different hash only produces a warning: onsemi updates its models.
