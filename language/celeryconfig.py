"""
Celery configuration for asynchronous task queue.
================================================

This module defines the configuration parameters used by Celery to
connect to a message broker and result backend.  By default the
application uses a local Redis instance running on the host
``redis`` at the standard port.  These settings can be overridden
with environment variables:

* ``CELERY_BROKER_URL`` – URL of the broker (e.g. ``redis://localhost:6379/0``).
* ``CELERY_RESULT_BACKEND`` – URL of the result store.

This file is deliberately lightweight; you can extend it with
additional options such as time limits, task routing, retry policy
and beat schedules as your deployment grows.  See the Celery
documentation for further details.
"""

import os

# Broker and backend URLs.  Fall back to Redis on the standard ports.
broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# Serialisers.  We restrict to JSON for portability; if you wish to
# transmit binary payloads such as audio you can switch to pickle
# (though be mindful of security implications).
task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

# Track when tasks start and finish; useful for monitoring.
task_track_started = True

# Optional: schedule periodic tasks here.  Initially empty.
beat_schedule: dict = {}
