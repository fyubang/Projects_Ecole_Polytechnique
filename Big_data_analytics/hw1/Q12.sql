select sname, pname, c.cost
from catalog c, parts p, suppliers s
where c.pid = p.pid
and c.sid = s.sid
and c.cost < (select avg(c2.cost)
from catalog c2
where c2.pid = p.pid
group by c2.pid);