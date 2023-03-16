import os
import sys
import glob
import time
import joblib
import traceback
from pathlib import Path
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from scipy import stats
import xgboost as xgb
from IPython.display import display
import matplotlib.pyplot as plt
import statsmodels.api as sm

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
from tqdm.notebook import tqdm
import timm

from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import Levenshtein
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import plotly.graph_objects as go
from IPython.display import HTML
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
import tifffile as tif
from PIL import Image, ImageEnhance;
from matplotlib import animation, rc;
import plotly
import PIL
import cv2
import plotly.io as pio
from learntools.core import binder
from learntools.game_ai.ex4 import *

from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, Optional, Tuple

from joblib import delayed, Parallel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm_notebook as tqdm

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error as msle
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt, style
from robot.trades import Trade
from td.client import TDClient

class OrderStatus():

    def __init__(self, trade_obj: Trade) -> None:

        self.trade_obj = trade_obj
        self.order_status = self.trade_obj.order_status

    @property
    def is_cancelled(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order was filled or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `FILLED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'FILLED':
            return True
        else:
            return False

    @property
    def is_rejected(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order was rejected or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `REJECTED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'REJECTED':
            return True
        else:
            return False

    @property
    def is_expired(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order has expired or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `EXPIRED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'EXPIRED':
            return True
        else:
            return False

    @property
    def is_replaced(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order has been replaced or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `REPLACED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'REPLACED':
            return True
        else:
            return False

    @property
    def is_working(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is working or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `WORKING`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'WORKING':
            return True
        else:
            return False

    @property
    def is_pending_activation(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is pending activation or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `PENDING_ACTIVATION`, 
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'PENDING_ACTIVATION':
            return True
        else:
            return False

    @property
    def is_pending_cancel(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is pending cancellation or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `PENDING_CANCEL`, 
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'PENDING_CANCEL':
            return True
        else:
            return False

    @property
    def is_pending_replace(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is pending replacement or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `PENDING_REPLACE`, 
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'PENDING_REPLACE':
            return True
        else:
            return False

    @property
    def is_queued(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is in the queue or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `QUEUED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'QUEUED':
            return True
        else:
            return False

    @property
    def is_accepted(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order was accepted or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `ACCEPTED`, `False`
            otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'ACCEPTED':
            return True
        else:
            return False

    @property
    def is_awaiting_parent_order(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is waiting for the parent order
        to execute or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `AWAITING_PARENT_ORDER`,
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'AWAITING_PARENT_ORDER':
            return True
        else:
            return False

    @property
    def is_awaiting_condition(self, refresh_order_info: bool = True) -> bool:
        """Specifies whether the order is waiting for the condition
        to execute or not.
        Arguments:
        ----
        refresh_order_info {bool} -- Specifies whether you want
            to refresh the order data from the TD API before 
            checking. If `True` a request will be made to the
            TD API to grab the latest Order Info.
        Returns
        -------
        bool
            `True` if the order status is `AWAITING_CONDITION`,
            `False` otherwise.
        """

        if refresh_order_info:
            self.trade_obj._update_order_status()

        if self.order_status == 'AWAITING_CONDITION':
            return True
        else:
            return False
        
def main():
  pass
