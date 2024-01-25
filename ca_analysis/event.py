import abc
import typing
from pathlib import Path
from typing import (
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import numpy as np
import pandas as pd
import yaml

from .common import *
from .io.common import *

__all__ = [
    "Event",
    "EventModel",
    "EventSchema",
    "EventSequence",
]


class EventBase(abc.ABC):
    """

    """

    _id: Optional[int]
    _name: Optional[str]
    _schema: Optional["EventSchema"]

    id = property(fget=lambda self: self._id)
    name = property(fget=lambda self: self._name)
    schema = property(fget=lambda self: self._schema)

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        schema: Optional["EventSchema"] = None,
    ):
        self._id = int(id) if id is not None else None
        self._name = str(name) if name is not None else None
        self._schema = schema

    def __int__(self) -> Optional[str]:
        return self._id

    def __str__(self) -> Optional[str]:
        return self._name


class Event(EventBase):
    """

    """

    _value: Optional[Number]
    _duration: Optional[Number]

    value = property(fget=lambda self: self._value)
    duration = property(fget=lambda self: self._duration)

    def __init__(
        self,
        id: int,
        name: str,
        value: Optional[Number] = None,
        duration: Optional[Number] = None,
        schema: Optional["EventSchema"] = None,
        **kw,
    ):
        super().__init__(id, name, schema)
        self._value = value
        self._duration = duration
        for key, val in kw.items():
            setattr(self, key, val)

    def __eq__(self, other: Union[int, str, "Event"]) -> bool:
        if is_int(other):
            return other == self.id
        if is_str(other):
            return other == self.name
        if isinstance(other, Event):
            return self.id == other.id and self.name == other.name
        return False

    def __repr__(self) -> str:
        s = "<Event: "
        s += f"name={self.name}, id={self.id}, value={self.value}, "
        s += f"duration={self.duration}>"
        return s


class EventSequence(EventBase):
    """
    Effectively, these are like sequence classes more than actual sequences.
    Each represents a kind of sequence (i.e., 'ABCD'), whereas an
    EventModel.Sequence object represents a single instance or occurrence
    of one of these sequence types.
    """

    _events: Optional[Tuple[Event, ...]]
    events = property(fget=lambda self: self._events)

    def __init__(
        self,
        id: Optional[Union[int, typing.Sequence]] = None,
        name: Optional[str] = None,
        events: Optional[typing.Sequence] = None,
        schema: Optional["EventSchema"] = None,
        **kw,
    ):
        super().__init__(id, name, schema)
        events = [] if events is None else list(events)
        if self._schema is not None:
            for i, elt in enumerate(events):
                if not isinstance(elt, Event):
                    events[i] = self._schema.get(event=elt)
        self._events = np.array(events, dtype=object)
        for key, val in kw.items():
            setattr(self, key, val)

    def astype(self, dtype: DTypeLike) -> NDArray:
        return self.events.astype(dtype)

    def __contains__(self, obj):
        return obj in self.events

    def __getitem__(self, key: int):
        return self.events[key]

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        s = "<EventSequence: "
        s += f"name={self.name}, id={self.id}, "
        s += f"events={tuple(self.events.astype(str))}>"
        return s


class EventSchema:

    _name: Optional[str] = None

    def __init__(self, url: Union[PathLike, URL]):
        self._url = URL(url)
        self._attrs = {}
        self._objects = {}
        self._tables = {}
        self._indexes = {}
        self._is_loaded = False

    @property
    def url(self) -> URL:
        return self._url

    @property
    def path(self) -> Path:
        return Path(self._url)

    @property
    def name(self) -> str:
        self._ensure_loaded()
        return self._name

    @property
    def description(self) -> str:
        self._ensure_loaded()
        return self._description

    @property
    def attrs(self) -> Mapping:
        self._ensure_loaded()
        return self._attrs

    @property
    def tables(self) -> Mapping:
        self._ensure_loaded()
        return self._tables

    @property
    def events(self) -> Tuple[Event]:
        self._ensure_loaded()
        return self._objects["events"]

    @property
    def sequences(self) -> Tuple[EventSequence]:
        self._ensure_loaded()
        return self._objects["sequences"]

    def get(self, **kw) -> EventBase:

        if "event" in kw:
            assert "sequence" not in kw
            return self.get_event(kw["event"])
        if "sequence" in kw:
            assert "event" not in kw
            return self.get_sequence(kw["sequence"])
        raise ValueError("invalid arguments")

    def get_event(self, event: [Union[int, str, Event]] = None) -> Event:
        self._ensure_loaded()
        if isinstance(event, Event):
            return event
        if is_int(event):
            return self._indexes["events"]["id"][event]
        return self._indexes["events"]["name"][event]

    def get_sequence(
        self,
        sequence: [Union[int, str, EventSequence]],
    ) -> EventSequence:
        self._ensure_loaded()
        if isinstance(sequence, EventSequence):
            return sequence
        if is_int(sequence):
            return self._indexes["sequences"]["id"][sequence]
        return self._indexes["sequences"]["name"][sequence]

    def load(self) -> None:

        data = self.load_yaml(self._url.path)
        self._name = data.pop("name", "")
        self._description = data.pop("description", "")

        self._tables = {
            "stimuli": None,
            "events": None,
            "sequences": None,
        }
        self._objects = {
            "stimuli": None,
            "events": [],
            "sequences": [],
        }
        self._indexes = {
            "stimuli": dict(id={}, name={}),
            "events": dict(id={}, name={}),
            "sequences": dict(id={}, name={}),
        }

        df = data.pop("stimuli", None)
        if df is not None:
            self._tables["stimuli"] = df

        df = data.pop("events", None)
        if df is not None:
            self._tables["events"] = df
            for i in range(len(df)):
                id, row = df.index[i], df.iloc[i].to_dict()
                name = row.pop("name")
                etc = {key: val for key, val in row.items() if is_str(key)}
                obj = Event(id, name, schema=self, **etc)
                self._objects["events"].append(obj)
                self._indexes["events"]["id"][id] = obj
                self._indexes["events"]["name"][name] = obj

        df = data.pop("sequences", None)
        if df is not None:
            self._tables["sequences"] = df
            for i in range(len(df)):
                id, row = df.index[i], df.iloc[i].to_dict()
                name = row.pop("name")
                event_ids = row.pop("event_ids")
                if "events" in row:
                    event_ids = row.pop("events")
                events = []
                for ev_id in event_ids:
                    events.append(self._indexes["events"]["id"][ev_id])
                obj = EventSequence(id, name, events=events, schema=self, **row)
                self._objects["sequences"].append(obj)
                self._indexes["sequences"]["id"][id] = obj
                self._indexes["sequences"]["name"][name] = obj

        self._objects["events"] = tuple(self._objects["events"])
        self._objects["sequences"] = tuple(self._objects["sequences"])
        self._attrs = dict(data)
        self._is_loaded = True

    @staticmethod
    def load_yaml(path: PathLike) -> None:

        def dict_to_dataframe(dct: Mapping) -> pd.DataFrame:

            # ensure dict has integer ids for keys and is in order.
            index = np.array(sorted([int(num) for num in dct.keys()]), dtype=int)
            if not np.array_equal(index, np.arange(len(index))):
                raise ValueError('invalid keys in yaml table (not 0, 1, ..., n - 1')
            dct = {key: dct[key] for key in index}

            # find fields
            fields = []
            for key, val in dct.items():
                fields.extend(list(val.keys()))
            fields = set(fields)

            # columnize
            col_data = {}
            for col_name in fields:
                vals = [row.get(col_name) for row in dct.values()]
                if any(isinstance(obj, list) for obj in vals):
                    vals = [np.array(obj) for obj in vals]
                col_data[col_name] = vals

            df = pd.DataFrame(col_data, index=index)
            df.index.name = "id"
            return df

        with open(path, "r") as f:
            raw = yaml.load(f, Loader=yaml.Loader)

        attrs = {k: v for k, v in raw.items() if not isinstance(v, dict)}
        dicts = {k: v for k, v in raw.items() if isinstance(v, dict)}

        stim_dict = dicts.get("stimuli", {})
        ev_dict = dicts.get("events", {})
        seq_dict = dicts.get("sequences", {})
        tables = {}

        if stim_dict:
            stim_df = dict_to_dataframe(stim_dict)
            stim_df.attrs["name"] = "event_schema.stimulus"
            tables["stimuli"] = stim_df
        if ev_dict:
            ev_df = dict_to_dataframe(ev_dict)
            ev_df.attrs["name"] = "event_schema.event"
            tables["events"] = ev_df
        if seq_dict:
            seq_df = dict_to_dataframe(seq_dict)
            seq_df.attrs["name"] = "event_schema.sequence"
            tables["sequences"] = seq_df

        data = attrs.copy()
        data.update(tables)
        return data

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def __getitem__(self, table: str) -> EventBase:
        return self.tables[table]

    def __repr__(self) -> str:
        self._ensure_loaded()
        s = "EventSchema"
        if self.name:
            s += f"({self.name})"
        s += "\n"
        for table_name, df in self._tables.items():
            if df is None:
                continue
            s += f"- Table('{table_name}')\n"
            dfs = "\n".join(["   " + ln for ln in repr(df).split("\n")])
            s += dfs + "\n\n"

        return s


class EventModel:
    """
    Interface to a session's event-related data.
    
    """

    def __init__(self, url: Union[PathLike, URL]):
        self._url = URL(url)
        self._schema = None
        self._tables = None
        self._is_loaded = False

    @property
    def url(self) -> URL:
        return self._url

    @property
    def path(self) -> Path:
        return Path(self._url)

    @property
    def schema(self) -> EventSchema:
        self._ensure_loaded()
        return self._schema

    @property
    def tables(self) -> Mapping[str, Optional[pd.DataFrame]]:
        self._ensure_loaded()
        return self._tables

    def get_fps(self) -> Number:
        """
        Compute framerate from timestamps.
        Returns
        -------

        """
        time = self.tables["frames"]["time"]
        return 1 / np.median(np.ediff1d(time))

    def load(self) -> None:

        self._schema = EventSchema(Path(self._url.path) / "schema.yaml")
        self._tables = {}
        for name in ("frames", "events", "sequences", "blocks"):
            path = Path(self._url.path) / f"{name}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, index_col=0)
            df.attrs["name"] = name
            self._tables[name] = df

        # add stimulus column to frames and events tables.
        schema = self._schema
        if 'stimuli' in schema.tables:
            if 'frames' in self._tables:
                frames = self._tables['frames']
                frames['stimulus'] = schema['events'].loc[frames.event.values].stimulus.values
            if 'events' in self._tables:
                events = self._tables['events']
                events['stimulus'] = schema['events'].loc[events.event.values].stimulus.values

        self._is_loaded = True

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def __getitem__(self, table: str) -> pd.DataFrame:
        self._ensure_loaded()
        try:
            return self.tables[table]
        except KeyError:
            if table in {'frame', 'event', 'sequence', 'block'}:
                msg = f"Deprecation warning: '{table}' is now {table + 's'}"
                warn(msg)
                return self.tables[table + 's']
            raise


