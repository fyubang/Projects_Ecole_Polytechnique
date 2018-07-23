select sname, count(distinct pid)
from suppliers s, catalog c
where s.sid = c.sid
group by s.sid;