select distinct c1.pid
from catalog c1, catalog c2
where c1.sid <> c2.sid
and c1.pid = c2.pid
order by pid;