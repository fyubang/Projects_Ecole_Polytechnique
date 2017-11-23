select distinct sname
from Catalog c, Suppliers s, Parts p
where c.sid = s.sid
and c.pid = p.pid
and color = 'Red'
and cost < 100
order by sname;