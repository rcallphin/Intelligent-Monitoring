from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import requests
import sys
import argparse
import time
import logging
import os
import pandas as pd

"""
A program to pull the result of a Prometheus query, calculate an apdex and find consistent patterns
between other metrics.
"""

class Stats_Exporter():

    def init(self):
        """
        Initializing class object for Prometheus query
        """

        self.logger = logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
        self.logger.debug("Starting exporter program.")


    def arg_setup(self):
        """
        Setup of command line arguments
        """

        timestamp_now = time.time()
        parser = argparse.ArgumentParser(description="Query the prometheus API and covert the metrics to csv.")
        parser.add_argument('-p', '--prom', required=True, help="Prometheus server to execute against include the port. (Usually 9090)")
        parser.add_argument('-q', '--query', required=True, help="Stat to query for, only one is allowed at a time.")
        parser.add_argument('-s', '--start', required=True, help="Start of time range for metric. Should be a unix timestamp.")
        parser.add_argument('-e', '--end', default=timestamp_now, help="End time range for metric. Should be a unix timestamp.")
        parser.add_argument('-a', '--apdex', nargs='+', help="Define satisfied, tolerated, sample and deprecated bounds for score. Space delimited list")
        parser.add_argument('-m', '--metrics', nargs='+', help="Define other metrics to compare against apdex")
        parser.add_argument('--step', default=15, help="Step range between datapoints.  Defaults to 15s.")
        try:
            self.args = vars(parser.parse_args())
        except:
            print('Usage: {0} -p http://prom:9090 -q node_boottime -e <endtime> -a 5 9 60'.format(sys.argv[0]))
            sys.exit(1)
        self.csv = str(self.args['query']) + ".csv"


    def collect(self):
        """
        Method to collect data from Promethesus and start processing of data
        """

        self.data = []
        response = self._request_data(self.args['query'], self.args['start'], self.args['end'], self.args['step'])
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException:
            if "exceeded maximum resolution" in response.json()['error']:
                self._create_chunk()
        finally:
            self._process_data_chunk()
            self._write_to_csv()
            self._gen_apdex()

    def _train_predict_lstm(self):

        tf.logging.set_verbosity(tf.logging.INFO)
        csv_file_name = "./" + self.csv
        reader = tf.contrib.timeseries.CSVReader(csv_file_name)
        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=16, window_size=16)
        with tf.Session() as sess:
            data = reader.read_full()
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            data = sess.run(data)
            coord.request_stop()

        ar = tf.contrib.timeseries.ARRegressor(
            periodicities=100, input_window_size=10, output_window_size=6,
            num_features=1,
            loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

        ar.train(input_fn=train_input_fn, steps=14000)

        evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
        evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

        (predictions,) = tuple(ar.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                evaluation, steps=250)))

        plt.figure(figsize=(15, 5))
        plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
        plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
        plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
        plt.xlabel('time_step')
        plt.ylabel('values')
        plt.legend(loc=4)
        plt.savefig(self.args['query'] + ".png")

    def _class_args(self, prom, query, start_time, end_time, step):
        """
        Setup of argumets for recursive call on class from apdex Method
        """

        self.args = {}
        self.args['prom'] = prom
        self.args['query'] = query
        self.args['start'] = start_time
        self.args['end'] = end_time
        self.args['step'] = step
        self.csv = str(self.args['query']) + ".csv"
        self.args['apdex'] = None


    def _create_chunk(self):
        """
        Pase out chucks of time for when timerange exceeds limits of Prometheus
        """

        # Half the size of the start and stop
        if "chunk" not in self.args:
            self.args['chunk'] = (int(self.args['end']) - int(self.args['start'])) / 2
            self.args['total_reduction'] = self.args['chunk']
        else:
            self.args['chunk'] = (int(self.args['chunk'])) / 2
            self.args['total_reduction'] = self.args['total_reduction'] + self.args['chunk']
        # Create new end time
        if self.args['total_reduction']:
            chunk_end = int(self.args['end']) - int(self.args['total_reduction'])
        else:
            chunk_end = int(self.args['end']) - int(self.args['chunk'])
        # Try requesting data with new end size.
        response = self._request_data(self.args['query'], self.args['start'], chunk_end, self.args['step'])
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException:
            if "exceeded maximum resolution" in response.json()['error']:
                self._create_chunk()
        return True
        # Check to see if the request was small enough


    def _create_labels(self, response):
        """
        Add lables for CSV output
        """

        results = response.json()['data']['result']
        for result in results:
            if len(result['metric'].keys()) > 0:
                self.labelnames = set([])
                self.labelnames.update(result['metric'].keys())
                # Canonicalize
                self.labelnames.discard('__name__')
                self.labelnames = sorted(self.labelnames)


    def _create_sample_size(self, df):
        """
        Break out requested sample size for apdex calculation
        """

        sample_size = int(self.args['apdex'][2])
        df_list = [df[i:i+sample_size] for i in range(0,df.shape[0],sample_size)]
        return df_list


    def _gen_apdex(self):
        """
        Generate apdex for requested metric, and initiate call for new metrics.
        (This method needs to be borken up)
        """

        if self.args['apdex'] is None:
            return True
        else:
            satisfied_limit = int(self.args['apdex'][0])
            tolerated_limit = int(self.args['apdex'][1])
            deprecated_apdex = float(self.args['apdex'][3])
            df = pd.read_csv(self.csv)
            self.csv = "apdex_" + self.csv
            df_list = self._create_sample_size(df)
            with open(self.csv, "w") as csv_file:
                wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
                index_count = 0
                #wr.writerow(['timestamp', 'apdex'])
                for data_sample in df_list:
                    satisfied_count = 0
                    tolerating_count = 0
                    time_stamp = data_sample['timestamp'].iloc[0]
                    for row in data_sample.itertuples(index=True):
                        if row[3] < satisfied_limit:
                            satisfied_count += 1
                        elif row[3] < tolerated_limit:
                            tolerating_count += 1
                    apdex = (float(satisfied_count) + float(tolerating_count)/2)/int(self.args['apdex'][2])
                    index_count += 1
                    row = [index_count, apdex]
                    wr.writerow(row)

            self._train_predict_lstm()
            self._recursive_collect()


    def _recursive_collect(self):

        metrics = self.args['metrics']
        for metric in metrics:
            get_metric = Stats_Exporter()
            get_metric._class_args(self.args['prom'], metric, self.args['start'], self.args['end'], self.args['step'])
            get_metric.collect()
            get_metric._normalize_csv()
            get_metric._train_predict_lstm()


    def _normalize_csv(self):

        df = pd.read_csv(self.csv)
        with open(self.csv, "w") as csv_file:
            wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            for index, row in df.iterrows():
                row = [index + 1, row[2]]
                wr.writerow(row)




    def _process_data_chunk(self, start=None):
        """
        Proccessing the chunked out results of query
        """

        # Recurively iterate using the chunk range
        if start is None:
            start = self.args['start']
        if "chunk" not in self.args.keys():
            end = int(self.args['end'])
        else:
            end = int(start) + int(self.args['chunk'])

        response = self._request_data(self.args['query'], start, end, self.args['step'])
        if 'labelnames' not in vars(self):
            self._create_labels(response)

        # Process the result
        try:
            results = response.json()['data']['result']
            for result in results:
                for value in result['values']:
                    metric = [result['metric'].get('__name__', '')] + value
                    for label in self.labelnames:
                        metric.append(result['metric'].get(label, ''))
                    self.data.append(metric)
        except requests.exceptions.RequestException:
            sys.exit(1)

        # Stop case - The end becomes the new start, so if the new start is older the end then stop.
        if int(end) >= int(self.args['end']):
            return True
        else:
            self._process_data_chunk(start=end)


    def _request_data(self, query, start, end, step):
        """
        Actual request to Prometheus
        """
        proxies = {'http': "socks5://127.0.0.1:7070"}
        response = requests.get('{0}/api/v1/query_range'.format(self.args['prom']),
                                ## Just for easy testing locally ##
                                proxies = proxies,
                                params={'query': query,
                                        'start': start,
                                        'end': end,
                                        'step': step})
        return response


    def _write_to_csv(self):
        """
        Writing to CST temporarily while we decide to store after prometheus
        """

        with open(self.csv, "w") as csv_file:
            wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            wr.writerow(['name', 'timestamp', 'value'] + list(self.labelnames))
            # Write the sanples.
            for row in self.data:
                wr.writerow(row)


# Execute the code
if __name__ == '__main__':
    exporter = Stats_Exporter()
    exporter.arg_setup()
    exporter.collect()
