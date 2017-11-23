select sname
from suppliers s
where not exists(select *
from parts p
where p.color = 'Red'
and not exists(select *
from catalog c
where s.sid=c.sid
and c.pid=p.pid))
order by sname;