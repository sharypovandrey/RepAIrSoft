from django.conf.urls import url
from app import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^qr_code_scan/', admin.site.urls),
    url(r'^get_schedule/', admin.site.urls),
    url(r'^add_log_item/', admin.site.urls),
    url(r'^push/', admin.site.urls),
]
