select pname, color
from Catalog c, Suppliers s, Parts p
where c.sid = s.sid
and c.pid = p.pid
and s.sname = 'Perfunctory Parts'
order by pname, color;