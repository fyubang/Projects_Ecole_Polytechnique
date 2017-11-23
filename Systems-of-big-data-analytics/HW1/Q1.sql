select distinct sname, address
from Catalog c, Suppliers s
where c.sid=s.sid
order by sname, address;