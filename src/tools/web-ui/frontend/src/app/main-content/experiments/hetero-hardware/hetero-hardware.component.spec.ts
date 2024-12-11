import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HeteroHardwareComponent } from './hetero-hardware.component';

describe('HeteroHardwareComponent', () => {
  let component: HeteroHardwareComponent;
  let fixture: ComponentFixture<HeteroHardwareComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HeteroHardwareComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HeteroHardwareComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
