# Data source provenance and limitations

- **Jolpica-F1** supplies historical calendar, race, qualifying and official result tables. Preserve its required attribution and non-commercial/share-alike terms as documented in this project.
- **OpenF1** supplies historical lap, stint, weather, pit-stop, starting-grid and race-control exports where available. Respect the upstream API's access terms and limits.
- **TIF1 0.9.0** supplies completed-session historical lap/sector/tyre data from public CDNs. TIF1 describes these as FastF1-derived files from TracingInsights repositories, with session data available after publication. The TIF1 library code is MIT-licensed and marked beta on PyPI; that does not automatically grant rights to redistribute all underlying Formula 1 timing data. Review the upstream data repository's terms before redistribution or commercial use.
- This project does not use direct FastF1 3.9+ authenticated session loading in its historical collectors. No F1 email or password should be entered into these commands.

References:
- https://pypi.org/project/tif1/
- https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/accounts_auth.rst
