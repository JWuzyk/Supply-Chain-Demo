# See FB Prophet

"""
Given historical data on various products, this model outputs a predicted demanded.
"""

import pandas as pd
from prophet import Prophet
import streamlit as st

def load_data(final_year=2018):
    #https://www.census.gov/retail/marts/www/timeseries.html
    #https://www.census.gov/retail/marts/www/adv44300.txt
    df = pd.read_csv('adv44300.tsv', header=1, delim_whitespace=True, nrows=30)
    df = df.melt(id_vars=['YEAR'], value_vars=df.columns[1:])
    df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df["variable"].astype(str))
    df = df[df['Date'] < pd.to_datetime(f'{final_year}-12-30', yearfirst=True)]
    return df[["Date", "value"]].rename(columns={"Date": "ds", "value": "y"})


class Forecaster:

    def fit_model(self, df, settings) -> object:
        m = Prophet(
            seasonality_mode=settings['seasonality_mode'],
            changepoint_prior_scale=settings['changepoint_prior_scale']
            )
        m.add_seasonality(
            name='yearly', period=7, fourier_order=10, prior_scale=10)

        m.fit(df)
        return m


class ProphetDisplay:

    def __init__(self, m, settings):
        self.model = m
        self.settings = settings

    def show(self):
        future = self.model.make_future_dataframe(periods=self.settings['forecast_length'], freq='MS')
        forecast = self.model.predict(future)
        fig1 = self.model.plot(forecast)
        st.write(fig1)


class Settings:

    def show(self):
        settings = {}
        settings['forecast_length'] = st.slider("Forcast Length")
        settings['seasonality_mode'] = st.selectbox("Seasonality Effect", options=["multiplicative",'additive'])
        settings['final_year'] = st.slider("Final Year", min_value=1994, max_value=2021, value=2019)
        settings['changepoint_prior_scale'] = st.slider("Change Point Prior Scale", value=0.5, max_value=20.0)
        return settings


class ForecastScreen:

    def show(self):
        st.markdown(
        """
        # Scenario: Forecasting
        ## Monthly US Retail Sales: Electronics and Appliance Stores

        """
        )
        settings_col, display_col = st.columns([5, 15])

        with settings_col:
            settings = Settings().show()

        model = Forecaster().fit_model(load_data(settings['final_year']),settings)

        with display_col:
            ProphetDisplay(model, settings).show()


def init_state():
    if "screen" not in st.session_state:
        st.session_state.screen = ForecastScreen()


if __name__ == "__main__":
    st.set_page_config(layout="wide")


    init_state()
    st.session_state.screen.show()