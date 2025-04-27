import boto3
import cv2
import os

def get_hls_stream_url():
    # Step 1: 取得 HLS URL
    kvs_client = boto3.client('kinesisvideo', region_name='us-west-2')

    endpoint = kvs_client.get_data_endpoint(
        StreamName='mirror',
        APIName='GET_HLS_STREAMING_SESSION_URL'
    )['DataEndpoint']

    kvs_archived_media_client = boto3.client('kinesis-video-archived-media',
                                            endpoint_url=endpoint,
                                            region_name='us-west-2')

    hls_stream = kvs_archived_media_client.get_hls_streaming_session_url(
        StreamName='mirror',
        PlaybackMode='LIVE'
    )

    hls_url = hls_stream['HLSStreamingSessionURL']

    return hls_url

if __name__ == "__main__":
    hls_url = get_hls_stream_url()
    print("HLS URL:", hls_url)
