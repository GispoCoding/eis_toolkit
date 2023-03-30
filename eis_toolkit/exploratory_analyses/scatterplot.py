from pathlib import Path

import pandas as pd
import plotly.express as px

# import rasterio

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../../tests/data/remote/small_raster_multiband.tif")

templates = [
    "ggplot2",
    "seaborn",
    "simple_white",
    "plotly",
    "plotly_white",
    "plotly_dark",
    "presentation",
    "xgridoff",
    "ygridoff",
    "gridon",
    "none",
]


def _df_plot_scatterplot(df, save_path=None, save_format="html"):
    fig = px.scatter(df, template="none")

    if save_path is not None:
        if save_format == "html":
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, save_format)

    fig.show()
    return fig


def _raster_plot_scatterplot(raster, save_path=None, save_format="html"):
    # df = pd.DataFrame()
    # for band, data in enumerate(raster.read()):
    #     df[f'band{band+1}'] = data.ravel()

    x = raster.read(1).ravel()
    y = raster.read(2).ravel()

    fig = px.scatter(x=x, y=y, template="none")

    if save_path is not None:
        if save_format == "html":
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, save_format)

    fig.show()
    return fig


def _raster_plot_histogram(raster, save_path=None, save_format="html"):
    data = raster.read()

    df1 = pd.DataFrame(data=data[0].ravel(), columns=["data"]).assign(band="1")
    df2 = pd.DataFrame(data=data[1].ravel(), columns=["data"]).assign(band="2")

    df = pd.concat([df1, df2])
    # for band, data in enumerate(raster.read()):
    #     df[f'band{band+1}'] = data.ravel()

    fig = px.histogram(df, template="none", animation_frame="band")

    if save_path is not None:
        if save_format == "html":
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, save_format)

    fig.show()
    return fig


def _parallel_coordinates_plot(df):
    fig = px.parallel_coordinates(
        df,
        color="species_id",
        # labels={"species_id": "Species",
        #         "sepal_width": "Sepal Width",
        #         "sepal_length": "Sepal Length",
        #         "petal_width": "Petal Width",
        #         "petal_length": "Petal Length", },
        color_continuous_scale=px.colors.sequential.haline,
        # color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2,
    )

    fig.show()
    return fig


# with rasterio.open(raster_path) as raster:
#     raster_plot_histogram(raster)
# raster_plot_scatterplot(raster)

_parallel_coordinates_plot(df=px.data.iris())
