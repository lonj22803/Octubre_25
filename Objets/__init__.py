"""
Objets package - Contains city, hotel, restaurant, and tourist place modules.
"""

from .City.City import CityBase
from .City.MetroSystem import MetroSystem
from .Hotel.Hotels import Hotels
from .Restaurant.Restaurants import Restaurants
from .TouristPlace.TouristPlace import TouristPlace

__all__ = [
    'CityBase',
    'MetroSystem',
    'Hotels',
    'Restaurants',
    'TouristPlace'
]
