"""Tests for the extra keyless data providers (ROADMAP section E).

Pure ``parse_*`` helpers and country resolution are exercised offline with small
sample payloads, so no network is needed.

Run:  backend/venv/bin/python backend/tests/test_datasources.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import datasources as D


def test_entity_candidates():
    assert D._entity_candidates("Spain snap general election 29 November 2026") == ["Spain"]
    assert D._entity_candidates("Vox party and Spain tension") == ["Vox", "Spain"]
    assert D._entity_candidates("Pedro Sánchez steps down") == ["Pedro Sánchez"]
    assert D._entity_candidates("a quiet summer") == []


def test_parse_wikidata():
    payload = {
        "search": [
            {
                "id": "Q1",
                "label": "Pedro Sánchez",
                "description": "Prime Minister of Spain since 2018",
                "concepturi": "http://www.wikidata.org/entity/Q1",
            }
        ]
    }
    out = D.parse_wikidata(payload)
    assert len(out) == 1
    assert out[0]["title"] == "Pedro Sánchez"
    assert out[0]["url"].startswith("http")
    assert "Prime Minister" in out[0]["snippet"]
    assert out[0]["source"] == "Wikidata"


def test_parse_wikidata_skips_incomplete_and_non_dict():
    assert D.parse_wikidata(None) == []
    payload = {"search": [{"label": ""}, {"label": "x"}, {"description": "no url"}]}
    assert D.parse_wikidata(payload) == []


def test_parse_gdelt():
    payload = {
        "articles": [
            {"title": "Spain heads to polls", "url": "https://a.example/x",
             "domain": "a.example", "seendate": "20261001T120000Z", "language": "English"}
        ]
    }
    out = D.parse_gdelt(payload)
    assert out[0]["title"] == "Spain heads to polls"
    assert out[0]["source"] == "GDELT"
    assert "a.example" in out[0]["snippet"]


def test_parse_gdelt_tolerates_error_body():
    assert D.parse_gdelt("Please limit requests to one every 5 seconds") == []
    assert D.parse_gdelt({"articles": [{"title": "no url"}]}) == []


def test_parse_worldbank():
    payload = [
        {"page": 1},
        [
            {"indicator": {"id": "SP.POP.TOTL", "value": "Population, total"},
             "country": {"value": "Spain"}, "countryiso3code": "ESP",
             "date": "2025", "value": 48797875},
            {"indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
             "country": {"value": "Spain"}, "countryiso3code": "ESP",
             "date": "2025", "value": 1906453309985.88},
            {"indicator": {"id": "FP.CPI.TOTL.ZG", "value": "Inflation"},
             "country": {"value": "Spain"}, "countryiso3code": "ESP",
             "date": "2024", "value": None},
        ],
    ]
    out = D.parse_worldbank(payload)
    assert len(out) == 2
    assert out[0]["title"] == "Spain · Population, total"
    assert out[0]["snippet"] == "48,797,875 (2025)"
    assert out[0]["source"] == "World Bank"
    assert "ESP" in out[0]["url"]
    assert "1,906,453,309,986" in out[1]["snippet"]


def test_parse_worldbank_malformed():
    assert D.parse_worldbank(None) == []
    assert D.parse_worldbank([]) == []
    assert D.parse_worldbank([{"page": 1}]) == []


def test_parse_imf_latest_year():
    payload = {"values": {"NGDP_RPCH": {"ESP": {"2024": 3.1, "2025": 2.3, "2026": 1.8}}}}
    out = D.parse_imf(payload, "ESP", "Spain")
    assert len(out) == 1
    assert out[0]["title"] == "Spain · real GDP growth"
    assert out[0]["snippet"] == "1.8% (2026)"
    assert out[0]["source"] == "IMF"


def test_parse_imf_missing():
    assert D.parse_imf({}, "ESP") == []
    assert D.parse_imf({"values": {"NGDP_RPCH": {}}}, "ESP") == []


def test_render_facts():
    lines = D.render_facts([
        {"title": "Spain · Population, total", "snippet": "48,797,875 (2025)", "source": "World Bank"},
        {"title": "", "snippet": "", "source": "IMF"},
    ])
    assert lines == ["WORLD BANK: Spain · Population, total 48,797,875 (2025)"]
    assert D.render_facts(None) == []


def test_parse_predictit_selects_and_formats_odds():
    payload = {
        "markets": [
            {
                "name": "Spanish general election 2026",
                "shortName": "Spain election winner",
                "url": "https://www.predictit.org/markets/detail/1/x",
                "status": "Open",
                "contracts": [
                    {"name": "Party A", "bestBuyYesCost": 0.42, "lastTradePrice": 0.4},
                    {"name": "Party B", "bestBuyYesCost": 0.31},
                ],
            },
            {"name": "Unrelated sports market", "shortName": "Final score", "status": "Open",
             "contracts": [{"name": "X", "bestBuyYesCost": 0.5}]},
        ]
    }
    out = D.parse_predictit(payload, "Spain election 2026")
    assert len(out) == 1
    assert "Party A" in out[0]["snippet"]
    assert "42% YES" in out[0]["snippet"]
    assert out[0]["source"].startswith("Prediction market")


def test_parse_predictit_no_overlap_returns_empty():
    payload = {"markets": [{"name": "Baseball championship", "shortName": "MLB",
                            "status": "Open", "contracts": [{"name": "X", "bestBuyYesCost": 0.5}]}]}
    assert D.parse_predictit(payload, "Spain election") == []


def test_parse_acled():
    payload = {
        "data": [
            {"event_type": "Protests", "location": "Madrid", "country": "Spain",
             "event_date": "2026-09-20", "fatalities": 0, "notes": "Mass demonstration downtown.",
             "source": "El País"},
        ]
    }
    out = D.parse_acled(payload)
    assert len(out) == 1
    assert out[0]["title"] == "Protests — Madrid, Spain"
    assert out[0]["source"] == "ACLED"
    assert "2026-09-20" in out[0]["snippet"]


def test_resolve_country_longest_match_wins():
    index = {"korea": "KOR", "south korea": "PRK", "spain": "ESP", "us": "USA"}
    assert D.resolve_country("tensions in South Korea escalate", index) == "PRK"
    assert D.resolve_country("the US midterm elections", index) == "USA"
    assert D.resolve_country("a quiet summer", index) is None
    assert D.resolve_country("", index) is None


def test_resolve_country_respects_word_boundaries():
    index = {"us": "USA"}
    assert D.resolve_country("jesus was born", index) is None


def test_acled_enabled_gating():
    key, email = os.environ.pop("ACLED_API_KEY", None), os.environ.pop("ACLED_EMAIL", None)
    try:
        assert D.acled_enabled() is False
        os.environ["ACLED_API_KEY"] = "k"
        assert D.acled_enabled() is False
        os.environ["ACLED_EMAIL"] = "e"
        assert D.acled_enabled() is True
    finally:
        os.environ.pop("ACLED_API_KEY", None)
        os.environ.pop("ACLED_EMAIL", None)
        if key is not None:
            os.environ["ACLED_API_KEY"] = key
        if email is not None:
            os.environ["ACLED_EMAIL"] = email


def test_fmt_number():
    assert D._fmt_number(100.0) == "100"
    assert D._fmt_number(2.0) == "2"
    assert D._fmt_number(48.123) == "48.1"
    assert D._fmt_number(1906453309985.88) == "1,906,453,309,986"


def test_tokens_drops_stopwords_and_short_words():
    toks = D._tokens("The US will hold general elections on the budget in 2026")
    assert "budget" in toks
    assert "the" not in toks and "will" not in toks and "us" not in toks
    assert "2026" not in toks and "elections" not in toks


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("test_")
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
