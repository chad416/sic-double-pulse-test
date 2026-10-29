# onsemi SPICE models (not included)

The onsemi EliteSiC M3S LTspice libraries are vendor-encrypted and carry the notice
"Permission is granted to use this file for simulations but not to reverse engineer
its contents". That grants use, not redistribution, so they are not committed here and
are listed in `.gitignore`.

Get them with:

```bash
python -m dpt.fetch_models
```

The script downloads both ZIPs from onsemi's public model URL into this folder, checks
their SHA256 against the versions used for the committed results (1200 V library v2.3,
Sep 2025; 650 V library v2.6, Mar 2026; hashes in `SHA256SUMS`) and unpacks them. A
different hash only produces a warning, because onsemi updates its models.

If the download is refused, save the ZIPs by hand from the product pages
(Technical Documentation, Simulation Models) and run the script again:

| File | Product page |
|---|---|
| `onsemi_sicmosfet_1200m3s_ltspice.zip` | https://www.onsemi.com/products/discrete-power-modules/silicon-carbide-sic/silicon-carbide-sic-mosfets/nth4l022n120m3s |
| `onsemi_sicmosfet_650m3s_ltspice.zip` | https://www.onsemi.com/products/discrete-power-modules/silicon-carbide-sic/silicon-carbide-sic-mosfets/nth4l032n065m3s |
