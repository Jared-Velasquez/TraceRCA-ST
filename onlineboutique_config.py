ENABLE_ALL_FEATURES = False

FEATURE_NAMES = ['latency', 'http_status']

FAULT_TYPES = {'cpu', 'delay', 'disk', 'loss', 'mem', 'socket'}

INVOLVED_SERVICES = [
    'adservice',
    'cartservice',
    'checkoutservice',
    'currencyservice',
    'emailservice',
    'frontendservice',
    'paymentservice',
    'productcatalogservice',
    'recommendationservice',
    'shippingservice',
]

SERVICE2IDX = {service: idx for idx, service in enumerate(INVOLVED_SERVICES)}
