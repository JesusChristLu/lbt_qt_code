from minio import Minio

client = Minio(
    endpoint="10.10.46.82:9000",
    access_key="admin",
    secret_key="bylz@2021",
    secure=False
)


def test_minio(bucket_name):
    print(client.list_buckets())
    client.make_bucket(bucket_name)
    print(client.list_buckets())
    client.remove_bucket(bucket_name)
    print(client.list_buckets())


def test_list(bucket_name):
    # print([x.object_name for x in client.list_objects(bucket_name=bucket_name,)])
    print(client.list_buckets())
    res = client.list_objects(bucket_name, recursive=True)
    # res = client.list_objects(bucket_name)
    for x in res:
        print(x.object_name)


if __name__ == '__main__':
    # test_minio("123123")
    test_list("mimi")
