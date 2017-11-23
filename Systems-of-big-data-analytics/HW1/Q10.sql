select p.pname, max(c.cost), avg(c.cost)
from parts p, catalog c
where p.pid = c.pid
group by p.pname
order by pname;