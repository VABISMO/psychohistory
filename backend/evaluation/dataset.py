"""A small labelled dataset of resolved socio-political events (ROADMAP section I).

Each record is a *binary question* with a known outcome, used to regression-test
and calibrate the engine:

    id           unique key
    question     the forecast question, phrased as a yes/no proposition
    category     coarse topic (used for base-rate priors)
    as_of        ISO date the forecast would be made
    deadline     ISO date by which the event must occur for outcome 1
    resolved_on  ISO date the event actually occurred (None if it did not)
    outcome      1 if it occurred within [as_of, deadline], else 0

Outcomes are public and well documented; ``resolved_on`` enables lead-time
scoring. This is deliberately small and conservative — extend it over time.
"""

DATASET = [
    # --- elections ---------------------------------------------------------
    dict(id="spain_2019", question="Spain holds a general election in 2019",
         category="elections", as_of="2019-01-01", deadline="2019-12-31",
         resolved_on="2019-11-10", outcome=1),
    dict(id="uk_2019", question="The United Kingdom holds a general election in 2019",
         category="elections", as_of="2019-01-01", deadline="2019-12-31",
         resolved_on="2019-12-12", outcome=1),
    dict(id="us_2020", question="The United States holds a presidential election in 2020",
         category="elections", as_of="2020-01-01", deadline="2020-12-31",
         resolved_on="2020-11-03", outcome=1),
    dict(id="france_2022", question="France holds a presidential election in 2022",
         category="elections", as_of="2022-01-01", deadline="2022-12-31",
         resolved_on="2022-04-24", outcome=1),
    dict(id="brazil_2022", question="Brazil holds a presidential election in 2022",
         category="elections", as_of="2022-01-01", deadline="2022-12-31",
         resolved_on="2022-10-30", outcome=1),
    dict(id="turkey_2023", question="Turkey holds a general election in 2023",
         category="elections", as_of="2023-01-01", deadline="2023-12-31",
         resolved_on="2023-05-14", outcome=1),
    dict(id="uk_2024", question="The United Kingdom holds a general election in 2024",
         category="elections", as_of="2024-01-01", deadline="2024-12-31",
         resolved_on="2024-07-04", outcome=1),
    dict(id="india_2024", question="India holds a general election in 2024",
         category="elections", as_of="2024-01-01", deadline="2024-12-31",
         resolved_on="2024-06-04", outcome=1),
    # --- EU / UK -----------------------------------------------------------
    dict(id="brexit_2020", question="The United Kingdom leaves the European Union by 31 January 2020",
         category="eu_uk", as_of="2019-01-01", deadline="2020-01-31",
         resolved_on="2020-01-31", outcome=1),
    dict(id="uk_2nd_ref_2020", question="The United Kingdom holds a second EU referendum by 2020",
         category="eu_uk", as_of="2019-01-01", deadline="2020-12-31",
         resolved_on=None, outcome=0),
    dict(id="scotref_2023", question="Scotland holds an independence referendum by 2023",
         category="eu_uk", as_of="2020-01-01", deadline="2023-12-31",
         resolved_on=None, outcome=0),
    # --- US politics -------------------------------------------------------
    dict(id="shutdown_2018", question="The US federal government shuts down in 2018",
         category="us_politics", as_of="2018-01-01", deadline="2018-12-31",
         resolved_on="2018-01-20", outcome=1),
    dict(id="trump_impeach_2019", question="The US House impeaches Donald Trump in 2019",
         category="us_politics", as_of="2019-01-01", deadline="2019-12-31",
         resolved_on="2019-12-18", outcome=1),
    dict(id="trump_convict_2020", question="The US Senate convicts Trump in the 2020 impeachment trial",
         category="us_politics", as_of="2020-01-01", deadline="2020-02-28",
         resolved_on=None, outcome=0),
    dict(id="tpp_ratify_2020", question="The United States ratifies the Trans-Pacific Partnership by 2020",
         category="us_politics", as_of="2017-01-01", deadline="2020-12-31",
         resolved_on=None, outcome=0),
    # --- geopolitics -------------------------------------------------------
    dict(id="ukraine_invasion_2022", question="Russia launches a full-scale invasion of Ukraine in 2022",
         category="geopolitics", as_of="2022-01-01", deadline="2022-12-31",
         resolved_on="2022-02-24", outcome=1),
    dict(id="nk_test_post2017", question="North Korea conducts a nuclear test after 2017",
         category="geopolitics", as_of="2018-01-01", deadline="2023-12-31",
         resolved_on=None, outcome=0),
    dict(id="iran_deal_2024", question="The United States and Iran reach a new nuclear agreement by 2024",
         category="geopolitics", as_of="2021-01-01", deadline="2024-12-31",
         resolved_on=None, outcome=0),
    # --- Middle East -------------------------------------------------------
    dict(id="abraham_2020", question="Israel signs normalization agreements with Arab states in 2020",
         category="middle_east", as_of="2020-01-01", deadline="2020-12-31",
         resolved_on="2020-09-15", outcome=1),
    dict(id="saudi_israel_2023", question="Saudi Arabia normalizes relations with Israel by 2023",
         category="middle_east", as_of="2021-01-01", deadline="2023-12-31",
         resolved_on=None, outcome=0),
    dict(id="gaza_war_2023", question="A major Israel–Hamas war erupts in 2023",
         category="middle_east", as_of="2023-01-01", deadline="2023-12-31",
         resolved_on="2023-10-07", outcome=1),
    dict(id="gaza_ceasefire_2024", question="A lasting Israel–Hamas ceasefire is reached in 2024",
         category="middle_east", as_of="2024-01-01", deadline="2024-12-31",
         resolved_on=None, outcome=0),
    # --- macro -------------------------------------------------------------
    dict(id="fed_hike_2022", question="The US Federal Reserve raises interest rates in 2022",
         category="macro", as_of="2022-01-01", deadline="2022-12-31",
         resolved_on="2022-03-16", outcome=1),
    dict(id="us_recession_2023", question="The United States enters a recession in 2023",
         category="macro", as_of="2023-01-01", deadline="2023-12-31",
         resolved_on=None, outcome=0),
    dict(id="btc_100k_2021", question="Bitcoin exceeds $100,000 by 2021",
         category="macro", as_of="2021-01-01", deadline="2021-12-31",
         resolved_on=None, outcome=0),
    dict(id="btc_100k_2024", question="Bitcoin exceeds $100,000 by 2024",
         category="macro", as_of="2024-01-01", deadline="2024-12-31",
         resolved_on="2024-12-04", outcome=1),
    # --- health ------------------------------------------------------------
    dict(id="covid_pandemic_2020", question="The WHO declares a pandemic in 2020",
         category="health", as_of="2020-01-01", deadline="2020-12-31",
         resolved_on="2020-03-11", outcome=1),
    dict(id="covid_vaccine_2020", question="A COVID-19 vaccine is authorised by the end of 2020",
         category="health", as_of="2020-01-01", deadline="2020-12-31",
         resolved_on="2020-12-02", outcome=1),
    # --- space / tech ------------------------------------------------------
    dict(id="spacex_crew_2020", question="SpaceX launches astronauts to the ISS in 2020",
         category="space", as_of="2020-01-01", deadline="2020-12-31",
         resolved_on="2020-05-30", outcome=1),
    dict(id="jwst_launch_2021", question="The James Webb Space Telescope launches in 2021",
         category="space", as_of="2021-01-01", deadline="2021-12-31",
         resolved_on="2021-12-25", outcome=1),
]


def by_id(records=None):
    records = DATASET if records is None else records
    return {r["id"]: r for r in records}
