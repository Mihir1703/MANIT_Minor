import os
import boto3
import timeit

ACCESS_ID = 'DO00QUPV42Q6RWQC3RAK'
SECRET_KEY = 'fKtjBTxyDNUSxKNaf61Uk245sqPjkLaGaMq1SCQm7BM'

session = boto3.session.Session()
client = session.client('s3',
                        region_name='nyc3',
                        endpoint_url='https://cloud-minor.nyc3.digitaloceanspaces.com',
                        aws_access_key_id=ACCESS_ID,
                        aws_secret_access_key=SECRET_KEY)


def upload_to_cloud(filepath, user):
    links = []
    if os.path.isdir(filepath):
        for i in os.listdir(filepath):
            client.upload_file(f'{filepath}/{i}', user, i)
            links.append(client.generate_presigned_url('get_object',
                                                       Params={'Bucket': user,
                                                               'Key': i},
                                                       ExpiresIn=60*60*24))
    else:
        client.upload_file(filepath, user, f'{os.path.basename(filepath)}')
        links.append(client.generate_presigned_url('get_object',
                                                       Params={'Bucket': user,
                                                               'Key': f'{os.path.basename(filepath)}'},
                                                       ExpiresIn=60*60*24))
    return links

import time
start_time = time.time()
data = upload_to_cloud('/home/mihir/coding/Minor/processed_files/spare_compressed','sparse_minor')
print("--- %s seconds ---" % (time.time() - start_time))
print(data)