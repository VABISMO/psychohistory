"""Historical recurring-event dates used to backtest the temporal model.

All dates are public and widely documented. Each series is a list of ISO dates
for a recurring socio-political event. Inter-arrival times (in years) are
derived by the backtest harness.
"""

# Spanish general (Congress) elections.
SPAIN_GENERAL = [
    "1977-06-15", "1979-03-01", "1982-10-28", "1986-06-22", "1989-10-29",
    "1993-06-06", "1996-03-03", "2000-03-12", "2004-03-14", "2008-03-09",
    "2011-11-20", "2015-12-20", "2016-06-26", "2019-04-28", "2019-11-10",
    "2023-07-23",
]

# United Kingdom general elections.
UK_GENERAL = [
    "1974-02-28", "1974-10-10", "1979-05-03", "1983-06-09", "1987-06-11",
    "1992-04-09", "1997-05-01", "2001-06-07", "2005-05-05", "2010-05-06",
    "2015-05-07", "2017-06-08", "2019-12-12", "2024-07-04",
]

# United States presidential elections.
US_PRESIDENTIAL = [
    "1964-11-03", "1968-11-05", "1972-11-07", "1976-11-02", "1980-11-04",
    "1984-11-06", "1988-11-08", "1992-11-03", "1996-11-05", "2000-11-07",
    "2004-11-02", "2008-11-04", "2012-11-06", "2016-11-08", "2020-11-03",
    "2024-11-05",
]

SERIES = {
    "spain_general": SPAIN_GENERAL,
    "uk_general": UK_GENERAL,
    "us_presidential": US_PRESIDENTIAL,
}
