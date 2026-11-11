# Third-party data sources (not redistributed here)
| Dataset | Use in this work | Licence / terms (as read on 2026-11-09) | Citation |
|---|---|---|---|
| **OpenACC** (EU Joint Research Centre) | Real car-following trajectories: lead-vehicle speed profiles, 10 Hz, 7 test campaigns (AstaZero, Casale, Cherasco, JRC low speed, One_vehicle_multiple_drivers, Vicolungo, ZalaZone) | JRC catalogue states access conditions "No limitations" and points to the European Commission reuse notice: reuse is authorised provided the source is acknowledged; not applicable to third-party IP. No CC licence is named on the catalogue page (a secondary website claims CC BY 4.0; we do not rely on that). | European Commission, Joint Research Centre (2026): Open ACC Database. doi:10.2905/JRC.KMH3D00. Paper: Makridis, Mattas, Anesiadou, Ciuffo, "OpenACC. An open database of car-following experiments to study the properties of commercial ACC systems," Transportation Research Part C 125 (2021) 103047, doi:10.1016/j.trc.2021.103047. |
| **BMW i3 real driving cycles** (TUM) | Fit and validate the vehicle/battery model (speed, current, voltage, aux power) | IEEE DataPort page lists "Creative Commons Attribution" (CC BY 4.0). The page also mentions a DataPort subscription for file access, so we do not redistribute the raw files. | Steinstraeter, Buberger, Trifonov, "Battery and Heating Data in Real Driving Cycles," IEEE DataPort, 2020, doi:10.21227/6jr9-5235. |
| **Renault Zoe, 29 months** | Trip-level cross-car checks of the earlier ML study (not used in the headline MPC results) | CC BY 4.0 (Zenodo record). | Morea, Alessandrini, Spadaro (2022), Dataset: Electric car parameters over 29 months (v1), Zenodo, doi:10.5281/zenodo.7033914. |
| Stanford EV driving/charging data | Optional only; not used for reported results | Mendeley Data (check terms before use) | doi:10.17632/7vdkzpnjgj.2 |

## What this package redistributes
Only derived material: 10 Hz lead/predecessor speed windows cut from OpenACC (`derived_data/*.npz`, `matlab_verification_windows/`), fold/window tables, fitted BMW i3 parameters and result tables. OpenACC-derived files carry the acknowledgement above. If a co-author or the JRC prefers, delete the `.npz`/window CSV files; all results can be regenerated from the downloaded raw data with the code.

## Download
- OpenACC: https://data.jrc.ec.europa.eu/dataset/9702c950-c80f-4d2f-982f-44d06ea0009f (doi:10.2905/JRC.KMH3D00)
- BMW i3: https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles (doi:10.21227/6jr9-5235)
- Zoe: https://zenodo.org/records/7033914
Place under `data/raw/openacc`, `data/raw/bmw_i3`, `data/raw/zoe` as used by the scripts.
