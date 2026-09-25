import unittest
from unittest import mock

from src.retrieval import fetch_alerts
from src.retrieval.fetch_alerts import alerts_context, format_alerts, get_alerts


def alert(park, category, title="t", url="https://nps.gov/x"):
    return {"parkCode": park, "category": category, "title": title,
            "description": "d", "url": url, "lastIndexedDate": "2026-09-20 14:03:11.0"}


def nps_response(alerts):
    return mock.Mock(json=mock.Mock(return_value={"data": alerts}), raise_for_status=mock.Mock())


class GetAlertsTests(unittest.TestCase):
    def setUp(self):
        fetch_alerts.cache.update(fetched_at=None, alerts=None)
        for target, value in [("API_KEY", "key"), ("BASE_URL", "https://nps.test")]:
            patcher = mock.patch.object(fetch_alerts, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.get = self.patch("src.retrieval.fetch_alerts.requests.get")
        self.clock = self.patch("src.retrieval.fetch_alerts.time.time")
        self.clock.return_value = 1000.0

    def patch(self, target):
        patcher = mock.patch(target)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_fetches_california_and_keeps_supported_parks(self):
        self.get.return_value = nps_response([alert("yose", "Caution"), alert("muwo", "Caution")])

        self.assertEqual(len(get_alerts(["yose", "muwo"])), 1)
        self.assertEqual(self.get.call_args.kwargs["params"]["stateCode"], "CA")

    def test_cache_hit_within_ttl(self):
        self.get.return_value = nps_response([alert("yose", "Caution")])
        get_alerts(["yose"])
        self.clock.return_value += fetch_alerts.CACHE_TTL_SECONDS - 1

        get_alerts(["yose"])

        self.get.assert_called_once()

    def test_refetch_after_ttl(self):
        self.get.return_value = nps_response([])
        get_alerts(["yose"])
        self.clock.return_value += fetch_alerts.CACHE_TTL_SECONDS

        get_alerts(["yose"])

        self.assertEqual(self.get.call_count, 2)

    def test_failure_returns_none_and_backs_off(self):
        self.get.side_effect = RuntimeError("timeout")

        self.assertIsNone(get_alerts(["yose"]))
        self.clock.return_value += fetch_alerts.FAILURE_BACKOFF_SECONDS - 1
        self.assertIsNone(get_alerts(["yose"]))
        self.get.assert_called_once()

        self.clock.return_value += 1
        self.get.side_effect = None
        self.get.return_value = nps_response([alert("yose", "Caution")])
        self.assertEqual(len(get_alerts(["yose"])), 1)

    def test_missing_key_is_a_failure(self):
        with mock.patch.object(fetch_alerts, "API_KEY", None):
            self.assertIsNone(get_alerts(["yose"]))
        self.get.assert_not_called()

    def test_filters_by_park_and_sorts_by_severity(self):
        self.get.return_value = nps_response([
            alert("yose", "Information"), alert("deva", "Park Closure"),
            alert("yose", "Park Closure"), alert("yose", "Caution"),
        ])

        result = get_alerts(["yose"])

        self.assertEqual([a["category"] for a in result], ["Park Closure", "Caution", "Information"])

    def test_alerts_context_cases(self):
        self.assertEqual(alerts_context([]), "")
        self.get.assert_not_called()

        self.get.return_value = nps_response([])
        self.assertEqual(alerts_context(["yose"]), "No active NPS alerts for Yosemite National Park.")

        fetch_alerts.cache.update(fetched_at=1000.0, alerts=None)
        self.assertIn("nps.gov/yose/planyourvisit/conditions.htm", alerts_context(["yose"]))


class FormatAlertsTests(unittest.TestCase):
    def test_format_includes_park_category_date_and_url(self):
        text = format_alerts([alert("yose", "Park Closure", title="Tioga Road closed")])

        self.assertIn("[Yosemite National Park] Park Closure: Tioga Road closed (updated 2026-09-20)", text)
        self.assertIn("More: https://nps.gov/x", text)

    def test_format_omits_more_line_without_url(self):
        self.assertNotIn("More:", format_alerts([alert("lavo", "Information", url="")]))


if __name__ == "__main__":
    unittest.main()
