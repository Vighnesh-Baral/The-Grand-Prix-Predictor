"""Client for the public Jolpica F1 API.

The Ergast-compatible API is used for stable historical race, qualifying and
standings records. The alpha API is an optional source for practice results;
its coverage/schema may change, so the parser is deliberately defensive.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import CACHE_DIR, JOLPICA_ALPHA_BASE_URL, JOLPICA_BASE_URL

LOGGER = logging.getLogger(__name__)


class JolpicaAPIError(RuntimeError):
    """Raised when Jolpica returns an unsuccessful or unusable response."""


class JolpicaClient:
    def __init__(
        self,
        cache_dir: Path | str = CACHE_DIR / "jolpica",
        timeout: int = 30,
        min_request_interval: float = 0.27,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_request_interval = min_request_interval
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "F1-Race-Intelligence/0.1 (educational ML project)",
            "Accept": "application/json",
        })
        retry = Retry(
            total=4,
            connect=3,
            read=3,
            status=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + urlencode(sorted((params or {}).items()))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _get_json(
        self,
        base_url: str,
        path: str,
        params: dict[str, Any] | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        cache_path = self._cache_path(url, params)
        if cache_path.exists() and not refresh:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                LOGGER.warning("Ignoring unreadable API cache file: %s", cache_path)

        wait = self.min_request_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise JolpicaAPIError(f"Jolpica request failed for {response.url if 'response' in locals() else url}: {exc}") from exc
        except ValueError as exc:
            raise JolpicaAPIError(f"Jolpica returned non-JSON content for {url}") from exc

        if not isinstance(payload, dict):
            raise JolpicaAPIError(f"Unexpected JSON response type from {url}: {type(payload).__name__}")
        try:
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Could not write API cache %s: %s", cache_path, exc)
        return payload

    @staticmethod
    def _get_mrdata(payload: dict[str, Any]) -> dict[str, Any]:
        return payload.get("MRData") or payload.get("mrdata") or {}

    def _fetch_ergast_collection(
        self,
        path: str,
        collection_key: str,
        refresh: bool = False,
        page_size: int = 100,
        nested_key: str | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self._get_json(
                JOLPICA_BASE_URL,
                f"{path.strip('/')}.json",
                {"limit": page_size, "offset": offset},
                refresh=refresh,
            )
            mrdata = self._get_mrdata(payload)
            collection: list[dict[str, Any]] = []
            # Most endpoints nest the collection inside a table object.
            for table in mrdata.values():
                if isinstance(table, dict) and isinstance(table.get(collection_key), list):
                    collection = table[collection_key]
                    break
            # Standings and less-common endpoints may use another wrapper.
            if not collection and isinstance(mrdata.get(collection_key), list):
                collection = mrdata[collection_key]
            records.extend(x for x in collection if isinstance(x, dict))
            try:
                total = int(mrdata.get("total", len(records)))
            except (TypeError, ValueError):
                total = len(records)
            page_count = sum(len(item.get(nested_key, [])) for item in collection) if nested_key else len(collection)
            if not collection or page_count == 0:
                break
            counted_records = sum(len(item.get(nested_key, [])) for item in records) if nested_key else len(records)
            if counted_records >= total or page_count < page_size:
                break
            offset += page_count
        return records

    def get_schedule(self, season: int, refresh: bool = False) -> list[dict[str, Any]]:
        return self._fetch_ergast_collection(f"{season}/races", "Races", refresh=refresh)

    @staticmethod
    def _expand_race_session_rows(races: list[dict[str, Any]], nested_key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for race in races:
            for item in race.get(nested_key, []) or []:
                if isinstance(item, dict):
                    flattened = dict(item)
                    flattened["race"] = race
                    rows.append(flattened)
        return rows

    def get_race_results(self, season: int, refresh: bool = False) -> list[dict[str, Any]]:
        races = self._fetch_ergast_collection(f"{season}/results", "Races", refresh=refresh, nested_key="Results")
        return self._expand_race_session_rows(races, "Results")

    def get_qualifying_results(self, season: int, refresh: bool = False) -> list[dict[str, Any]]:
        races = self._fetch_ergast_collection(f"{season}/qualifying", "Races", refresh=refresh, nested_key="QualifyingResults")
        return self._expand_race_session_rows(races, "QualifyingResults")

    def get_sprint_results(self, season: int, refresh: bool = False) -> list[dict[str, Any]]:
        try:
            races = self._fetch_ergast_collection(f"{season}/sprint", "Races", refresh=refresh, nested_key="SprintResults")
            return self._expand_race_session_rows(races, "SprintResults")
        except JolpicaAPIError as exc:
            LOGGER.info("Sprint data unavailable for %s: %s", season, exc)
            return []

    def _get_standings(self, season: int, table_key: str, round_number: int | None = None, refresh: bool = False) -> list[dict[str, Any]]:
        path = f"{season}/{round_number}/" if round_number is not None else f"{season}/"
        payload = self._get_json(
            JOLPICA_BASE_URL,
            f"{path}{table_key}.json",
            {"limit": 100, "offset": 0},
            refresh=refresh,
        )
        mrdata = self._get_mrdata(payload)
        table = mrdata.get("StandingsTable", {})
        lists = table.get("StandingsLists", []) if isinstance(table, dict) else []
        if not lists:
            return []
        list_key = "DriverStandings" if table_key == "driverstandings" else "ConstructorStandings"
        return lists[0].get(list_key, [])

    def get_driver_standings(self, season: int, round_number: int | None = None, refresh: bool = False) -> list[dict[str, Any]]:
        return self._get_standings(season, "driverstandings", round_number, refresh)

    def get_constructor_standings(self, season: int, round_number: int | None = None, refresh: bool = False) -> list[dict[str, Any]]:
        return self._get_standings(season, "constructorstandings", round_number, refresh)

    @staticmethod
    def _alpha_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data", payload)
        if isinstance(data, dict):
            events = data.get("events", [])
            if isinstance(events, list):
                return [e for e in events if isinstance(e, dict)]
        return []

    @staticmethod
    def _append_practice_rows(
        rows: list[dict[str, Any]],
        result_items: list[dict[str, Any]],
        season: int,
        round_number: int | None,
        event_name: str,
        circuit_obj: dict[str, Any],
        event: dict[str, Any],
        session_code: str,
    ) -> None:
        for item in result_items:
            if not isinstance(item, dict):
                continue
            driver = item.get("driver") or item.get("Driver") or {}
            team = item.get("team") or item.get("constructor") or item.get("Constructor") or {}
            if not isinstance(driver, dict):
                driver = {"name": str(driver)}
            if not isinstance(team, dict):
                team = {"name": str(team)}
            driver_name = (
                driver.get("name")
                or " ".join(filter(None, [driver.get("given_name"), driver.get("family_name")]))
                or driver.get("full_name")
                or item.get("driver_name")
                or "Unknown"
            )
            team_name = team.get("name") or team.get("team_name") or item.get("team_name") or "Unknown"
            position = item.get("position") or item.get("position_number") or item.get("rank")
            raw_time = item.get("time") or item.get("fastest_time") or item.get("best_lap_time")
            time_value = (raw_time.get("value") or raw_time.get("display")) if isinstance(raw_time, dict) else raw_time
            rows.append({
                "season": season,
                "round": round_number,
                "race_name": event_name,
                "circuit_id": circuit_obj.get("id") or event.get("circuit_id"),
                "session": session_code,
                "position": position,
                "driver_id": driver.get("ergast_id") or driver.get("driver_id") or driver.get("id"),
                "driver_code": driver.get("code") or driver.get("abbreviation"),
                "driver_name": driver_name,
                "constructor_id": team.get("ergast_id") or team.get("team_id") or team.get("id"),
                "constructor_name": team_name,
                "time": time_value,
                "laps": item.get("laps") or item.get("number_of_laps"),
                "status": item.get("status"),
                "source": "Jolpica alpha",
            })

    def get_practice_results(self, season: int, refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch Jolpica alpha FP1/FP2/FP3 classifications where published.

        The alpha endpoint is intentionally optional: coverage is being backfilled
        and the response schema can evolve. Failures for an individual session do
        not invalidate the stable race/qualifying dataset.
        """
        try:
            schedule_payload = self._get_json(
                JOLPICA_ALPHA_BASE_URL,
                f"schedules/{season}/",
                refresh=refresh,
            )
        except JolpicaAPIError as exc:
            LOGGER.warning("Practice schedule unavailable for %s: %s", season, exc)
            return []

        rows: list[dict[str, Any]] = []
        for event in self._alpha_events(schedule_payload):
            round_obj = event.get("round") if isinstance(event.get("round"), dict) else {}
            round_id = round_obj.get("id") or event.get("round_id")
            if not round_id and str(event.get("id", "")).startswith("round_"):
                round_id = event.get("id")
            if not round_id:
                continue
            round_raw = (round_obj.get("number") or round_obj.get("round") or
                         event.get("round_number") or event.get("number"))
            try:
                round_number = int(round_raw)
            except (TypeError, ValueError):
                round_number = None
            event_name = (event.get("name") or event.get("event_name") or
                          round_obj.get("name") or round_obj.get("race_name") or "Unknown Grand Prix")
            circuit_obj = event.get("circuit") if isinstance(event.get("circuit"), dict) else {}
            event_row_count = len(rows)
            session_codes = ("FP1", "FP2", "FP3")
            for session_code in session_codes:
                try:
                    result_payload = self._get_json(
                        JOLPICA_ALPHA_BASE_URL,
                        f"results/{round_id}/{session_code}/",
                        refresh=refresh,
                    )
                except JolpicaAPIError as exc:
                    LOGGER.debug("%s %s %s unavailable: %s", season, event_name, session_code, exc)
                    continue
                data = result_payload.get("data", result_payload)
                result_items = data.get("results", []) if isinstance(data, dict) else []
                if not isinstance(result_items, list):
                    continue
                self._append_practice_rows(
                    rows, result_items, season, round_number, event_name, circuit_obj, event, session_code
                )

            # Some alpha API event payloads expose a combined FP classification
            # rather than individual FP1/FP2/FP3 endpoints. Try that only when no
            # separate practice session was returned for this event.
            if len(rows) == event_row_count:
                try:
                    result_payload = self._get_json(
                        JOLPICA_ALPHA_BASE_URL,
                        f"results/{round_id}/FP/",
                        refresh=refresh,
                    )
                except JolpicaAPIError as exc:
                    LOGGER.debug("%s %s combined FP unavailable: %s", season, event_name, exc)
                else:
                    data = result_payload.get("data", result_payload)
                    result_items = data.get("results", []) if isinstance(data, dict) else []
                    if isinstance(result_items, list):
                        self._append_practice_rows(
                            rows, result_items, season, round_number, event_name, circuit_obj, event, "FP"
                        )
        return rows
