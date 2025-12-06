#!/usr/bin/env python3

# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import deque
from dataclasses import dataclass

import lcm

from dimos.protocol.service.lcmservice import LCMConfig, LCMService


@dataclass
class LCMSpyConfig(LCMConfig): ...


class Topic:
    def __init__(self, name: str):
        self.name = name
        # Store (timestamp, data_size) tuples for statistics
        self.message_history = deque()

    def msg(self, data: bytes):
        print(f"> msg {self.__str__()} {len(data)} bytes")
        datalen = len(data)
        # Store timestamp and data size
        self.message_history.append((time.time(), datalen))
        # Keep only recent messages (clean up old entries)
        self._cleanup_old_messages()

    def _cleanup_old_messages(self, max_age: float = 60.0):
        """Remove messages older than max_age seconds"""
        current_time = time.time()
        while self.message_history and current_time - self.message_history[0][0] > max_age:
            self.message_history.popleft()

    def _get_messages_in_window(self, time_window: float):
        """Get messages within the specified time window"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        return [(ts, size) for ts, size in self.message_history if ts >= cutoff_time]

    # avg msg freq in the last n seconds
    def freq(self, time_window: float) -> float:
        messages = self._get_messages_in_window(time_window)
        if not messages:
            return 0.0
        return len(messages) / time_window

    # avg kbps in the last n seconds
    def kbps(self, time_window: float) -> float:
        messages = self._get_messages_in_window(time_window)
        if not messages:
            return 0.0
        total_bytes = sum(size for _, size in messages)
        total_kbits = (total_bytes * 8) / 1000  # Convert bytes to kbits
        return total_kbits / time_window

    # avg msg size in the last n seconds
    def size(self, time_window: float) -> float:
        messages = self._get_messages_in_window(time_window)
        if not messages:
            return 0.0
        total_size = sum(size for _, size in messages)
        return total_size / len(messages)

    def __str__(self):
        return f"topic({self.name})"


class LCMSpy(LCMService):
    default_config = LCMSpyConfig
    topic = dict[str, Topic]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.topic = {}
        self.l = lcm.LCM(self.config.url) if self.config.url else lcm.LCM()

    def start(self):
        super().start()
        self.l.subscribe("/.*", self.handle_message)

    def stop(self):
        """Stop the LCM spy and clean up resources"""
        if hasattr(self, "l") and self.l:
            self.l.close()
        super().stop()

    def handle_message(self, topic, data):
        if topic not in self.topic:
            self.topic[topic] = Topic(topic)
        self.topic[topic].msg(data)


if __name__ == "__main__":
    lcm_spy = LCMSpy()
    lcm_spy.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("LCM Spy stopped.")
