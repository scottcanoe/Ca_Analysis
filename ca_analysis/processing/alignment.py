import warnings
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ca_analysis.common import *
from ca_analysis.event import EventSchema

__all__ = [
    "SignalParser",
]


class SignalParser:
    """
    Align imaging frames with thorsync's stimulus and hardware control signals.

    The purpose of this class is to associate each imaging frame with the
    stimulus that was onscreen. Also groups contiguous frames into 'events' and
    determiens sequences of events.

    by parsing "sync_data" (i.e., the DataFrame
    containing frame captures events, analog signals, strobes, etc.). Two
    dataframes are produced (*strobes* and *frames*) and saved as
    attributes of the StimulusAlignmentTool instance that processed the data.
    When processing is initiated, the supplied sync data will be copied
    and saved as the *sync_data* attribute, possibly modified.


    Parameters
    ----------

    schema : EventSchema
        Definitions of events/sequences.
    sync_data: pandas.DataFrame
       Input data. Should contain columns "time", "event_value", "frame_out",
       "frame_trigger", and  and "strobe".


    Notes
    -----

    Sinces strobes typically exist for many clocks, associating a single
    time (and therefore event value) with a strobe means choosing where in the
    strobe cycle we should use as its main point. The `strobe_anchor` attribute
    controls whether to use the rising edge ('start'), the falling edge ('stop'),
    or midpoint ('mid', the default). Similarly, frames exist for many cycles,
    and therefore the `frame_anchor` attribute plays the same role as for strobes.

    Processing the sync data creates the following tables (as attributes).

    * strobes: Contains locations of strobes in the input data and their associated
      events and voltage data. It contains the following columns.

      * start (int): Indices where strobe cycle began.
      * stop (int): Indices index where strobe cycle ended.
      * mid (int): Midpoints between start and stop indices.
      * time (float): Corresponding time (in seconds) for each strobe.
      * event_value (float): Corresponding value from `event_value` line.
      * event (int): Event associated with the event value.

    * frames: Containing locations of frames, it's nearest preceding strobe, etc.
      It has the following columns:

      * start (int): Sync index at which the frame capture started.
      * stop (int): Sync index at which the frame capture completed.
      * mid (int): Midpoints between start and stop indices.
      * time (float): Time corresponding to the middle of the frame.
      * event_value (float) : Analog event level when frame capture began.
      * event (int): ID of corresponding event.

    * events: Contains locations of events in frame indices and corresponding
      events in the event schema.

      * start (int): Frame index corresponding to beginning of this event.
      * stop (int): Frame index corresponding to last frame + 1.
      * event (int): ID of corresponding event in the schema.

    * sequences: Contains locations of sequences in event occurrence indices.

      * start (int): Frame index corresponding to beginning of this event.
      * stop (int): Frame index corresponding to last frame + 1.
      * sequence (int): ID of corresponding event in the schema.

    """

    version = "0.0.1"

    #: Each cycle in a square wave-like signal has a duration, but we want to
    #: collapse the cycle down to a single point in time where we will take a
    #: snapshot of whatever values are present in the other lines. Whether to
    #: use the rising edge, the falling edge, or somewhere in between a matter
    #: of the way the stimulus code was written.
    #:
    #  Example 1: we are using an analog signal to encode for one of a
    #: handful of discrete stimuli. Since there is some (albeit small) amount
    #: of time for an analog signal to transition between values, Jeff wrote
    #: his code to send the analog signal to the next value as soon as the
    #: strobe fires, and it spends most of its time hovering around the
    #: next value. This means we can't just read off the analog event value
    #: corresponding to each frame capture time. We have to capture the
    #: analog event value as soon as the strobe appears and assign that event
    #: value to all frames captured between that strobe and the next. In this
    #: case, it's best set `strobe_anchor` to 'start', thereby capturing the
    #: event value before it starts moving towards its next target. Setting
    #: `frame_anchor` to 'mid' minimizes slippage and misclassification of
    #: frames the begin capturing within a few milliseconds before the strobe.
    #:
    #: Example 2: if logging event values that encode for continuous variables,
    #: such as the horizontal position of a stimulus on a screen, it's
    #: better to
    #: just in time for the next event's strobe. Additionally, some experiments
    #: use continuous analog values to indicate, for example, the position
    #: of a bar across the screen during retinotopic mapping. In this case,
    #: we set `frame_anchor` to 'mid', and  which means we capture the event code
    #:
    #: When to look at time and event signals when associating events with
    #: strobes and assigning timestamps for strobes. When using Jeff's
    #: stimulus stuff, set this to 'start'. For most of my stuff, set to 'mid'.
    strobe_anchor: str = "start"

    #: When to look at time and event signals when associating events with
    #: strobes and assigning timestamps for frames.
    frame_anchor: str = "mid"

    def __init__(self, schema: EventSchema, sync_data: pd.DataFrame):

        self._schema = schema
        self._sync_data = sync_data
        self._tables = {}
        self._prepared = False

    @property
    def schema(self) -> EventSchema:
        return self._schema

    @property
    def sync_data(self) -> pd.DataFrame:
        return self._sync_data

    @property
    def tables(self) -> Mapping[str, pd.DataFrame]:
        return self._tables

    # -------------------------------------------------------------------------#
    #                              Public Methods                              #
    # -------------------------------------------------------------------------#

    def parse(self) -> None:

        self.prepare()
        self.parse_strobes()
        self.parse_frames()
        self.parse_events()
        self.parse_sequences()

    def prepare(self) -> None:
        """
        Prepare

        Preparation
        -----------

        - frame_out
          1. Ensure dtype is np.int8.
          2. Set the first and last elements to zero.

        - frame_trigger
          1. Ensure dtype in np.int8.
          2. Set the first and last elements to zero.
          3. If line goes up and down more than once, raise
             a ValueError.

        - strobe
          1. Ensure dtype is np.int8.
          2. Set the first and last elements to zero.
          3. Paste a short starting strobe at elements 1-5. This is needed
             for associating a frame with its nearest preceding strobe.

        """

        sd = self.sync_data.copy()

        name_map = {
            "AnalogEvents": "event_value",
            "FrameOut": "frame_out",
            "FrameTrigger": "frame_trigger",
            "Strobe": "strobe",
        }
        for key, val in name_map.items():
            if key in sd:
                sd[val] = sd[key]
                del sd[key]

        # - frame out
        if "frame_out" in sd:
            frame_out = sd["frame_out"].values.astype(np.int8)
            frame_out[0] = 0
            frame_out[-1] = 0
            sd["frame_out"] = frame_out

        # - frame trigger: not really in use at the moment.
        if "frame_trigger" in sd:
            frame_trigger = sd["frame_trigger"].values.astype(np.int8)
            frame_trigger[0] = 0
            frame_trigger[-1] = 0
            ft_cycles = find_cycles(frame_trigger)
            n_start, n_stop = len(ft_cycles["start"]), len(ft_cycles["stop"])
            if n_start == 0 and n_stop == 0:
                warnings.warn("'frame_trigger' appears to be unused")
            elif n_start > 1 or n_stop > 1:
                pass
                # raise ValueError(f"'frame_trigger' triggered {n_start} times.")
            sd["frame_trigger"] = frame_trigger

        # - strobe
        if "strobe" in sd:
            strobe = sd["strobe"].values.astype(np.int8)
            strobe[0] = 0
            strobe[-1] = 0
            strobe_cycles = find_cycles(strobe)
            n_start, n_stop = len(strobe_cycles["start"]), len(
                strobe_cycles["stop"]
            )
            if n_start == 0 and n_stop == 0:
                warnings.warn("'strobe' signal appears to be unused")
            sd["strobe"] = strobe

        self._sync_data = sd
        self._prepared = True

    def parse_strobes(self) -> None:
        """
        Creates and sets self.strobes if sync data uses strobes.
        As per the prepare method, the strobe line is guaranteed to
        be of type np.int8 and begin with a zero.

        """

        # Prepare for processing.
        if not self._prepared:
            self.prepare()

        if "strobe" not in self.sync_data:
            print("No strobe column found. Skipping `parse_strobes()`.")
            return

        # Get indices of strobe's rising edges, and get the event value at those
        # time points capture the event value (assuming there is an event
        # value line).
        sd = self.sync_data
        cycles = find_cycles(sd["strobe"])
        strobe_df = pd.DataFrame(
            {
                "start": cycles["start"],
                "stop": cycles["stop"],
                "mid": cycles["mid"],
                "time": sd["time"].values[cycles[self.strobe_anchor]],
            }
        )
        if "event_value" in sd:
            event_values = sd["event_value"].values[cycles[self.strobe_anchor]]
            strobe_df["event_value"] = event_values

            # Classify strobes according their event values.
            events = self.schema.events
            target_values = np.array([ev.value for ev in events])
            event_ids = np.zeros(len(strobe_df), dtype=int)
            for i, v in enumerate(event_values):
                ix_nearest = np.argmin(np.abs(v - target_values))
                ev = events[ix_nearest]
                event_ids[i] = ev.id
            strobe_df["event"] = event_ids

        self._tables["strobe"] = strobe_df

    def parse_frames(self) -> None:
        """
        Creates and sets self.frames if sync data uses 'FrameOut'.
        """

        # Prepare for processing.

        if "frame_out" not in self.sync_data:
            print("No 'frame_out' column found. Skipping parse_frames().")
            return
        if "strobe" not in self.tables:
            self.parse_strobes()

        # As with strobes, find the edges of the frame out signals, and
        # capture the indices and event values at the specified edge/midpoint.
        sd = self.sync_data
        cycles = find_cycles(sd["frame_out"].values)
        frame_df = pd.DataFrame(
            {
                "start": cycles["start"],
                "stop": cycles["stop"],
                "mid": cycles["mid"],
                "time": sd["time"].values[cycles[self.frame_anchor]],
            }
        )

        if "event_value" in sd:
            event_values = sd["event_value"].values[cycles[self.frame_anchor]]
            frame_df["event_value"] = event_values

            # If "event_value" signal was used, add 'event_value' column containing
            # the voltage value at the frame start/stop/mid points. Then find the
            # nearest preceding strobe, and associate its event id/name with the
            # frame itself.

            # Now associate each frame with the nearest preceding strobe
            # since that strobe may actually have the correct event value,
            # depending on how the stimulus code was written. Use the event
            # assigned to the previous strobe to as the event the frame
            # was classified as.
            if "strobe" in sd:
                strobe_df = self.tables["strobe"]
                strobe_pts = strobe_df[self.strobe_anchor].values
                frame_pts = frame_df[self.frame_anchor].values
                event_ids = np.zeros(len(frame_df), dtype=int)
                event_strobes = np.zeros(len(frame_df), dtype=int)
                for i, pt in enumerate(frame_pts):
                    n_preceding = np.sum(strobe_pts <= pt)
                    strobe_id = max(n_preceding - 1, 0)
                    row = strobe_df.iloc[strobe_id]
                    event_ids[i] = row["event"]
                    event_strobes[i] = strobe_id

                frame_df["event"] = event_ids
                frame_df["event_strobe"] = event_strobes

        self._tables["frame"] = frame_df

    def parse_events(self) -> None:
        """
        Group together frames by their assignment to the strobe indicating
        an event change.

        Record the start/stop indices of each contiguous block of same-event
        frames and which event was occurring. This will be the 'event' table.
        """
        """
        replace with find_repeats() in indexing?
        """
        schema = self.schema
        if "events" not in schema.tables or len(schema.tables["events"]) == 0:
            return

        # Prepare for processing if we haven't already.
        if "frame" not in self._tables:
            self.parse_frames()

        frame_df = self._tables["frame"]
        ev_strobes = frame_df["event_strobe"].values
        # replace the following with
        # np.r_[[0], np.argwhere(np.ediff1d(ev_strobes)).reshape(-1)]] ??
        starts = np.hstack([[0],
                            np.squeeze(np.argwhere(np.ediff1d(ev_strobes))) + 1
                            ]
                           )
        stops = np.hstack([starts[1:], [len(frame_df)]])
        event_ids = frame_df["event"].values[starts]

        event_df = pd.DataFrame({
            "event": event_ids,
            "start": starts,
            "stop": stops,
        })

        self._tables["event"] = event_df

    def parse_sequences(self) -> None:
        """
        Match sequences in the schema against the event table. Record
        the start and stop indices (into the event table) for each sequence
        presentation.


        """

        schema = self.schema
        if "sequences" not in schema.tables or \
                schema.tables["sequences"] is None or \
                len(schema.tables["sequences"]) == 0:
            return

        # Prepare for processing.
        if "event" not in self.tables:
            self.parse_events()
        event_df = self.tables["event"]

        # Put sequences in order from largest to smallest in length to capture
        # the longest token. Also make sure it's a tuple for equality
        # testing.

        # efficiency step: associate each sequence with constituent event ids.
        sequences = self.schema.sequences
        seq_events = {}
        for seq in sequences:
            seq_events[seq.name] = np.array([ev.id for ev in seq])

        # Now find sequences using element names as tokens.
        cols = {"start": [], "stop": [], "sequence": []}
        event_ids = event_df["event"].values
        i = 0
        while i < len(event_df):
            # At each point in the token stream, loop through all sequences
            # and try for a match. Accept the first one.
            accepted = False
            for seq in sequences:
                seq_len = len(seq)
                try:
                    if np.array_equal(event_ids[i:i + seq_len], seq_events[seq.name]):
                        cols["start"].append(i)
                        cols["stop"].append(i + seq_len)
                        cols["sequence"].append(seq.id)
                        i += seq_len
                        accepted = True
                        break
                except IndexError:
                    pass

            if not accepted:
                i += 1

        sequence_df = pd.DataFrame(cols)
        self._tables["sequence"] = sequence_df

    # --------------------------------------------------------------------------#
    # Saving, plotting, etc.

    def show(
        self,
        tlim: Optional[Tuple[Number, Number]] = None,
        ylim: Optional[Tuple[Number, Number]] = None,
        label_strobes: bool = False,
        label_frames: bool = False,
        event_values: bool = True,
        frame_trigger: bool = True,
        frame_out: bool = True,
        strobe: bool = True,
        downsample: Optional[int] = None,
    ):
        """
        Visualize the *sync_data*, *strobe_info*, and *frame_info*.
        """

        import matplotlib.pyplot as plt

        sd = self.sync_data
        tables = self.tables

        if tlim:
            X = sd["time"].values
            in_bounds = np.logical_and(X >= tlim[0], X <= tlim[1])
            del X

        def get_sd_XY(colname: str) -> Tuple[np.ndarray, np.ndarray]:
            X, Y = sd["time"].values, sd[colname].values
            if tlim:
                X, Y = X[in_bounds], Y[in_bounds]
            if downsample:
                X, Y = X[::downsample], Y[::downsample]
            return X, Y

        fig = plt.figure(figsize=(18, 4))
        ax = fig.add_subplot(1, 1, 1)

        # Draw frame_trigger.
        if frame_trigger and "FrameTrigger" in sd:
            X, Y = get_sd_XY("FrameTrigger")
            ax.step(X,
                    Y,
                    c="cyan",
                    alpha=0.5,
                    ls="-",
                    lw=3,
                    label="FrameTrigger"
                    )

        # Draw strobes.
        if strobe and "Strobe" in sd:
            X, Y = get_sd_XY("Strobe")
            ax.step(X, Y, c="red", alpha=0.5, label="Strobe")

        # Draw event values.
        if event_values and "AnalogEvents" in sd:
            X, Y = get_sd_XY("AnalogEvents")
            ax.plot(X,
                    Y,
                    c="blue",
                    alpha=0.5,
                    ls="-",
                    lw=1,
                    label="AnalogEvents"
                    )

        # Draw frame_out.
        if frame_out and "Strobe" in sd:
            X, Y = get_sd_XY("FrameOut")
            ax.step(X, Y, c="black", alpha=0.5, label="FrameOut")

        # Label strobes
        if label_strobes and "strobe" in tables:
            sd_time = sd["time"].values
            strobe_df = tables["strobe"]
            for i in range(len(strobe_df) - 1):
                row = strobe_df.iloc[i]
                if tlim:
                    if sd_time[row.start] < tlim[0] or sd_time[
                        row.start] > tlim[1]:
                        continue
                ax.text(
                    sd_time[row.mid],
                    1.0,
                    "{}".format(row.name),
                    c="red",
                    fontsize=8,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    color="red",
                )

        # Label frames
        if label_frames and "frame" in tables:
            sd_time = sd["time"].values
            frame_df = tables["frame"]
            for i in range(len(frame_df) - 1):
                row = frame_df.iloc[i]
                if tlim:
                    if row.time < tlim[0] or row.time > tlim[1]:
                        continue
                ax.text(
                    row.time,
                    1.0,
                    "{}".format(row.name),
                    c="black",
                    fontsize=8,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        if ylim:
            ax.set_ylim(ylim)

        # Add labels and legends.
        ax.set_xlabel("time (sec)")
        ax.legend()

        plt.tight_layout()
        plt.show()
        return fig


def find_cycles(arr: ArrayLike) -> dict:
    """
    Returns a dictionary containing indices corresponding to the start, stop, and midpoint
    of all cycles (i.e., a pulse in a binary signal).
    - start: indices where a cycle started
    - stop: indices where a cycle stopped
    - mid: center of cycle (computed as the median of start and stop indices)

    Parameters
    ----------
    arr: array-like
        1d array-like object of binary values (True/False of 0/1).

    Returns
    -------
    info: dict
      Has the following keys/values:
        - start: indices where a cycle started
        - stop: indices where a cycle stopped
        - mid: center of cycle (computed as the median of start and stop indices)

    """
    diffed = np.ediff1d(arr, to_begin=0)
    ixs = np.where(diffed)[0]
    signs = diffed[ixs]
    if len(signs) == 0:
        start = np.array([], dtype=int)
        stop = np.array([], dtype=int)
        mid = np.array([], dtype=int)
    elif len(signs) == 1:
        if signs[0] > 0:
            start = np.array([ixs[0]], dtype=int)
            stop = np.array([], dtype=int)
            mid = np.array([], dtype=int)
        else:
            start = np.array([], dtype=int)
            stop = np.array([ixs[0]], dtype=int)
            mid = np.array([], dtype=int)
    else:
        if signs[0] > 0:
            start = ixs[::2]
            stop = ixs[1::2]
        else:
            start = ixs[1::2]
            stop = ixs[::2]
        mid = np.median(np.vstack([start, stop]), axis=0).astype(int)

    info = {
        "start": start,
        "stop": stop,
        "mid": mid,
    }

    return info


