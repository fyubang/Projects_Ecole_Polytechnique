select distinct sname
from suppliers s, catalog c
where s.sid = c.sid
and c.sid not in(select c.sid
from catalog c
where c.cost>=100);