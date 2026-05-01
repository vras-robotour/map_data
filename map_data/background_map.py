import os
from typing import List
import requests
from PIL import Image
from geopy.distance import geodesic


URL_PREF = "https://maps.geoapify.com/v1/staticmap?style=osm-carto"


def get_url(w: int, h: int, corners: List[float]) -> str:
    """
    Get url for the background map.
    """
    api_key = os.environ.get("GEOAPIFY_API_KEY", "")
    if not api_key:
        api_key = "8f3be3c0c8484eceb15b0f50218c8c02"

    url = URL_PREF
    w_url = f"&width={w}"
    h_url = f"&height={h}"
    area_url = f"&area=rect:{corners[0]},{corners[1]},{corners[2]},{corners[3]}"
    api_url = f"&apiKey={api_key}"

    url = url + w_url + h_url + area_url + api_url

    return url


def get_background_image(
    min_long: float,
    max_long: float,
    min_lat: float,
    max_lat: float,
    x_margin: float,
    y_margin: float,
) -> Image.Image:
    """
    Get background image of the area.
    """
    width_m = geodesic(
        (max_lat + y_margin, max_long + x_margin),
        (min_lat - y_margin, max_long + x_margin),
    )
    height_m = geodesic(
        (max_lat + y_margin, max_long + x_margin),
        (max_lat + y_margin, min_long - x_margin),
    )
    ratio = width_m / height_m

    size_limit = min(4000, 150 * min(width_m.meters, height_m.meters))
    width = int(size_limit)
    height = int(width * ratio)

    while (
        width > size_limit or height > size_limit
    ):  # anything more can lead to wrongly cut background image (maybe only with large ratios)
        width = int(width * 0.9)
        height = int(width * ratio)

    corners = [
        min_long - x_margin,
        max_lat + y_margin,
        max_long + x_margin,
        min_lat - y_margin,
    ]
    url = get_url(width, height, corners)

    bg_map = Image.open(requests.get(url, stream=True).raw)

    return bg_map
