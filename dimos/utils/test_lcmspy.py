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

import pytest

from dimos.protocol.pubsub.lcmpubsub import PickleLCM, Topic
from dimos.utils.lcmspy import LCMSpy


def test_spy_basic():
    lcm = PickleLCM(autoconf=True)
    lcm.start()

    lcmspy = LCMSpy()
    lcmspy.start()

    video_topic = Topic(topic="/video")
    odom_topic = Topic(topic="/odom")

    for i in range(5):
        lcm.publish(video_topic, f"video frame {i}")
        time.sleep(0.1)
        if i % 2 == 0:
            lcm.publish(odom_topic, f"odometry data {i / 2}")

    # Wait a bit for messages to be processed
    time.sleep(0.5)

    # Test statistics for video topic
    video_topic_spy = lcmspy.topic["/video"]
    assert video_topic_spy is not None

    # Test frequency (should be around 10 Hz for 5 messages in ~0.5 seconds)
    freq = video_topic_spy.freq(1.0)
    assert freq > 0
    print(f"Video topic frequency: {freq:.2f} Hz")

    # Test bandwidth
    kbps = video_topic_spy.kbps(1.0)
    assert kbps > 0
    print(f"Video topic bandwidth: {kbps:.2f} kbps")

    # Test average message size
    avg_size = video_topic_spy.size(1.0)
    assert avg_size > 0
    print(f"Video topic average message size: {avg_size:.2f} bytes")

    # Test statistics for odom topic
    odom_topic_spy = lcmspy.topic["/odom"]
    assert odom_topic_spy is not None

    freq = odom_topic_spy.freq(1.0)
    assert freq > 0
    print(f"Odom topic frequency: {freq:.2f} Hz")

    kbps = odom_topic_spy.kbps(1.0)
    assert kbps > 0
    print(f"Odom topic bandwidth: {kbps:.2f} kbps")

    avg_size = odom_topic_spy.size(1.0)
    assert avg_size > 0
    print(f"Odom topic average message size: {avg_size:.2f} bytes")

    print(f"Video topic: {video_topic_spy}")
    print(f"Odom topic: {odom_topic_spy}")


def test_topic_statistics_direct():
    """Test Topic statistics directly without LCM"""
    from dimos.utils.lcmspy import Topic as TopicSpy

    topic = TopicSpy("/test")

    # Add some test messages
    test_data = [b"small", b"medium sized message", b"very long message for testing purposes"]

    for i, data in enumerate(test_data):
        topic.msg(data)
        time.sleep(0.1)  # Simulate time passing

    # Test statistics over 1 second window
    freq = topic.freq(1.0)
    kbps = topic.kbps(1.0)
    avg_size = topic.size(1.0)

    assert freq > 0
    assert kbps > 0
    assert avg_size > 0

    print(f"Direct test - Frequency: {freq:.2f} Hz")
    print(f"Direct test - Bandwidth: {kbps:.2f} kbps")
    print(f"Direct test - Avg size: {avg_size:.2f} bytes")


def test_topic_cleanup():
    """Test that old messages are properly cleaned up"""
    from dimos.utils.lcmspy import Topic as TopicSpy

    topic = TopicSpy("/test")

    # Add a message
    topic.msg(b"test message")
    initial_count = len(topic.message_history)
    assert initial_count == 1

    # Simulate time passing by manually adding old timestamps
    old_time = time.time() - 70  # 70 seconds ago
    topic.message_history.appendleft((old_time, 10))

    # Trigger cleanup
    topic._cleanup_old_messages(max_age=60.0)

    # Should only have the recent message
    assert len(topic.message_history) == 1
    assert topic.message_history[0][0] > time.time() - 10  # Recent message
