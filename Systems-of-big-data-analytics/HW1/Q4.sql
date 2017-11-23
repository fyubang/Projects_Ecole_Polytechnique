select distinct sname
from Catalog c, Suppliers s, Parts p
where c.sid = s.sid
and c.pid = p.pid
and cost < 100
and color = 'Red'
or color = 'Green'
order by sname;