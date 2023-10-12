import collections
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any, Iterable

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm.auto import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_KEY, PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.utils import batchify, load_cache


def unpack_scores(response_json: dict) -> Optional[Tuple[dict, dict]]:
    if not response_json:
        return None

    attribute_scores = response_json['attributeScores'].items()

    summary_scores = {}
    span_scores = {}
    for attribute, scores in attribute_scores:
        attribute = attribute.lower()

        # Save summary score
        assert scores['summaryScore']['type'] == 'PROBABILITY'
        summary_scores[attribute] = scores['summaryScore']['value']

        # Save span scores
        for span_score_dict in scores['spanScores']:
            assert span_score_dict['score']['type'] == 'PROBABILITY'
            span = (span_score_dict['begin'], span_score_dict['end'])
            span_scores.setdefault(span, {})[attribute] = span_score_dict['score']['value']

    return summary_scores, span_scores


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        if response['response']:
            response = unpack_scores(response['response'])[0]
        else:
            # import ipdb
            # ipdb.set_trace()
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


class PerspectiveAPI:
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 25):
        self.service = self._make_service(api_key)
        self.last_request_time = -1  # satisfies initial condition
        self.rate_limit = rate_limit
        self.next_uid = 0

    def request(self, texts: Union[str, List[str]]) -> List[Tuple[Optional[Dict[str, Any]], Optional[HttpError]]]:
        print(f"\n[Parallel thread] Start processing a request consisting on a batch of {len(texts)} generations.")
        if isinstance(texts, str):
            texts = [texts]

        # Rate limit to 1 batch request per second (60 requests per minute), a batch may include several generations (e.g, 2*25=50)
        assert len(texts) <= self.rate_limit
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
            print("[Parallel thread] I'm here waiting a bit as the last request was made less than a second ago!")
        self.last_request_time = time.time()

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in range(self.next_uid, self.next_uid + len(texts))}
        self.next_uid += len(texts)

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        print("[Parallel thread] Explicitly making the new batch htpp request.")
        batch_request = self.service.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(self._make_request(text, self.service), callback=response_callback, request_id=uid)
            print(f"[Parallel thread] Adding request {uid} to the batch request.")
        batch_request.execute()
        print("[Parallel thread] Explicitly executing the new batch htpp request.")

        return list(responses.values())
    
    def request_bulk(self,
                     corpus: Union[Iterable[str], Iterable[Tuple[str, str]]],
                     output_file: Union[str, Path],
                     pbar: tqdm = None):
        # Check for output file
        output_file = Path(output_file)
        assert not output_file.exists()

        # Set up progress bar
        if not pbar:
            total = len(corpus) if isinstance(corpus, collections.abc.Sequence) else None
            pbar = tqdm(total=total, dynamic_ncols=True)
        pbar.set_description(f'Perspective API')

        i = 0
        num_failures = 0
        print(f"[Parallel thread] Going to process a bulk of requests associated to a corpus of generations of length {len(list(corpus))}.")
        with output_file.open('a') as f:
            print("[Parallel thread] Output file opened correctly.")
            for batch in batchify(corpus, self.rate_limit):
                print("[Parallel thread] Processing first batch!")
                #print(f"[Parallel thread] Processing a batch of {len(batch)} samples...")
                request_ids = None
                if isinstance(batch[0], tuple):
                    request_ids, batch = zip(*batch)
                print("[Parallel thread] Calling the request() method...")
                for j, (response, exception) in enumerate(self.request(batch)):
                    response_dict = {
                        'request_id': request_ids[j] if request_ids else i,
                        'response': response,
                        'error': str(exception) if exception else None
                    }

                    # Save response
                    json.dump(response_dict, f)
                    f.write('\n')

                    if exception:
                        num_failures += 1
                        print("[Parallel thread] There was a HTTP failure here!")
                print("[Parallel thread] Batch processed!")
                i += len(batch)
                pbar.update(len(batch))
                pbar.set_postfix(failures=num_failures, rate_limt=self.rate_limit)

    @staticmethod
    def _make_service(api_key: str):
        # Generate API client object dynamically based on service name and version
        return discovery.build('commentanalyzer', 'v1alpha1',
                               discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                               developerKey=api_key,
                               static_discovery=False)

    @staticmethod
    def _make_request(text: str, service):
        analyze_request = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
            'spanAnnotations': True,
        }
        return service.comments().analyze(body=analyze_request)


class PerspectiveWorker:
    SENTINEL = 'STOP'

    def __init__(self, out_file: Path, total: int, rate_limit: int):
        if not rate_limit:
            print("Disabling Perspective API (rps is 0)")
            self.enabled = False
            return
        self.enabled = True

        self.requests_handled = set()
        for response in load_cache(out_file):
            self.requests_handled.add(response['request_id'])
        total -= len(self.requests_handled)

        # Setup worker thread
        self.task_queue = mp.Queue()
        self.process = mp.Process(target=self.perspective_worker,
                                  args=(self.task_queue, out_file, total, rate_limit))
        self.process.start()

    def __call__(self, request_id: str, text: str):
        if not self.enabled:
            return

        if request_id not in self.requests_handled:
            print(f"[Main thread] Putting request {request_id} to the queue to be further processed later.")
            self.task_queue.put((request_id, text))
            
    def stop(self):
        if not self.enabled:
            return

        print("[Main thread] Waiting for Perspective to finish...")
        self.task_queue.put(self.SENTINEL)
        self.process.join()

    @classmethod
    def perspective_worker(cls, queue: mp.Queue, responses_file: Path, total: int, rate_limit: int):
        queue_iter = iter(queue.get, cls.SENTINEL)
        api = PerspectiveAPI(rate_limit=rate_limit)
        pbar = tqdm(total=total, dynamic_ncols=True, position=1)
        api.request_bulk(queue_iter, output_file=responses_file, pbar=pbar)


def test_perspective_api():
    api = PerspectiveAPI()

    text_success = "Testing"
    text_error = 'x' * (20480 + 1)

    score_1, error_1 = api.request(text_success)[0]
    assert score_1 and not error_1

    score_2, error_2 = api.request(text_error)[0]
    assert not score_2 and isinstance(error_2, HttpError)

    multi_score, multi_error = zip(*api.request([text_success, text_error]))
    assert multi_score == (score_1, score_2)
    assert tuple(map(str, multi_error)) == tuple(map(str, (error_1, error_2)))


# test_perspective_api()
