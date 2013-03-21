import Queue
import threading
import urllib2

def geturl(q, url):
    q.put(urllib2.urlopen(url).read())

theurls = '''http://example.com/
             http://googel.com/
             http://example.co.uk'''.split()

q = Queue.Queue()

for u in theurls:
    t = threading.Thread(target=geturl, args=(q, u))
    t.daemon = True
    t.start()

s = q.get()
print s